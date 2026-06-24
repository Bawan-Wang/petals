# Petals Learning Note

## 1. 這個 repo 是做什麼的？

`petals` 是一個讓超大型語言模型能以分散式方式執行的 Python 套件。

它的核心想法不是把整個模型放在單一機器上，而是把模型的不同 Transformer blocks 分散到網路上的多個節點。client 端在推論時，會透過 P2P 網路找到持有這些 blocks 的 server，依序把 activations 傳過去執行，最後組成完整的前向推論或生成流程。

簡單說，Petals 想解決的是：

- 單機 GPU 放不下 70B、180B、405B 這種超大模型
- 但如果把每台機器只分擔幾層模型，就能共同提供推論能力
- 對使用者來說，操作方式仍盡量接近 Hugging Face Transformers

README 的定位很明確：

- 支援分散式 inference
- 支援 distributed fine-tuning / prompt tuning
- 可用 public swarm，也可自建 private swarm

## 2. Repo 的高層架構

這個專案可以拆成 5 個主要區塊：

### 2.1 Client

位置：`src/petals/client/`

用途：

- 提供使用者可直接呼叫的 distributed model 介面
- 管理遠端推論 session
- 將推論請求分配到遠端 server blocks

重點檔案：

- `src/petals/client/from_pretrained.py`
- `src/petals/client/inference_session.py`
- `src/petals/client/remote_generation.py`
- `src/petals/client/remote_sequential.py`

其中 `from_pretrained.py` 很重要，因為它把使用者熟悉的 Hugging Face 載入體驗延伸到 Petals 的 distributed model。

### 2.2 Routing

位置：`src/petals/client/routing/`

用途：

- 從 DHT 查出哪些節點持有哪些 blocks
- 根據延遲與吞吐量等資訊決定走哪一條 server 路徑
- 在多個候選節點之間做路由選擇

重點檔案：

- `src/petals/client/routing/sequence_manager.py`
- `src/petals/client/routing/sequence_info.py`
- `src/petals/client/routing/spending_policy.py`

`RemoteSequenceManager` 是 routing 的核心。它會維護目前可用的遠端 block 資訊，並在需要時組出一條可走的 sequence。

### 2.3 Server

位置：`src/petals/server/`

用途：

- 啟動一個 Petals 節點
- 載入模型的一部分 blocks
- 對外宣告自己持有的 blocks
- 接受 client 的 forward / backward / generation 請求

重點檔案：

- `src/petals/server/server.py`
- `src/petals/server/backend.py`
- `src/petals/server/handler.py`
- `src/petals/server/block_selection.py`
- `src/petals/server/from_pretrained.py`

`Server` 類別是整個節點生命週期的中樞，從模型設定讀取、裝置選擇、量化、attention cache、DHT 註冊、吞吐量估測，到 block rebalance 都在這裡處理。

### 2.4 Model Adapters

位置：`src/petals/models/`

用途：

- 針對不同模型家族包一層 Petals 的 distributed 實作
- 把 Hugging Face config / model type 映射到正確的 Petals 類別

目前看到的支援模型：

- BLOOM
- Falcon
- Llama
- Mixtral

每個子資料夾都會把自己的 config 與 model 類別註冊到自動選型系統。

### 2.5 CLI / Tooling

位置：`src/petals/cli/`

用途：

- 啟動 server
- 啟動 DHT/bootstrap node
- 提供部署與運維入口

重點檔案：

- `src/petals/cli/run_server.py`
- `src/petals/cli/run_dht.py`

## 3. 核心技術概念

### 3.1 DHT

Petals 使用 `hivemind` 的 DHT 做節點與 block 資訊發現。

可以把它理解成一個分散式索引：

- 哪個 peer 提供哪些 layers
- 這些 layers 的狀態如何
- 哪些節點目前可用

client 不需要事先知道每個 server 的地址，而是先查 DHT，再組路徑。

### 3.2 P2P 推論

推論不是把完整權重抓到本地，而是：

1. client 載入模型 config
2. client 查 DHT 找出一串可覆蓋全部 layers 的 server
3. hidden states / activations 逐段傳給遠端 server
4. 每個 server 只計算自己持有的 blocks
5. 結果一路傳遞直到整個模型完成

這就是 README 所說的 BitTorrent-style large model execution。

### 3.3 Auto model dispatch

`src/petals/utils/auto_config.py` 是這個 repo 很關鍵的 glue layer。

它的工作是：

- 先透過 Hugging Face `AutoConfig` 讀出 model type
- 再查 Petals 內部的 class mapping
- 選出對應的 distributed config / model class

這讓使用者能用類似下面的方式載入：

```python
from petals import AutoDistributedModelForCausalLM

model = AutoDistributedModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-405B-Instruct"
)
```

對外介面很像 Transformers，但內部已經切換成分散式執行。

## 4. 主要執行流程

### 4.1 使用者端流程

1. 呼叫 `AutoDistributedModelForCausalLM.from_pretrained(...)`
2. `auto_config.py` 依 model type 選出對應的 Petals 類別
3. 建立 client 端 distributed model
4. 透過 routing 查 DHT，找出可用的 block sequence
5. 進入遠端推論 / generation session
6. 依序呼叫遠端 server 完成完整 forward path

### 4.2 Server 端流程

1. 執行 `python -m petals.cli.run_server <model>`
2. `run_server.py` 解析參數
3. 建立 `Server` 物件
4. `Server` 載入模型 config
5. 決定要持有哪些 blocks
6. 載入 block weights，設定 dtype / quantization / cache
7. 啟動 DHT 與 request handlers
8. 定期向 DHT 宣告自己目前提供的 blocks

## 5. 值得先讀的程式

如果目標是「快速看懂 repo」，建議順序如下：

### 第一輪：先掌握整體用途

1. `README.md`
2. `setup.cfg`
3. `src/petals/__init__.py`

原因：

- README 告訴你產品定位與使用方式
- `setup.cfg` 告訴你依賴與版本限制
- `__init__.py` 告訴你 package 對外公開哪些 API

### 第二輪：看 server 啟動主線

1. `src/petals/cli/run_server.py`
2. `src/petals/server/server.py`

原因：

- `run_server.py` 是 CLI 入口
- `server.py` 是核心 orchestration

### 第三輪：看 client 推論主線

1. `src/petals/utils/auto_config.py`
2. `src/petals/client/from_pretrained.py`
3. `src/petals/client/routing/sequence_manager.py`
4. `src/petals/client/remote_generation.py`

原因：

- 這幾個檔案串起使用者最常見的載入與推論流程

### 第四輪：看模型適配層

1. `src/petals/models/__init__.py`
2. `src/petals/models/llama/__init__.py`
3. 需要時再看其他 model family

原因：

- 先理解 registration 機制，再進到單一模型族的 block/model 實作

## 6. 重要依賴與工程特徵

從 `setup.cfg` 可以看出幾個很重要的工程訊號：

- `transformers==4.43.1`
- `hivemind` 直接鎖定 git commit
- 使用 `tensor_parallel`
- 使用 `bitsandbytes`
- 有 `peft` 支援
- `numpy<2`

這代表：

- 專案高度依賴特定上游版本
- 相容性是重要議題
- 量化、平行化、PEFT 都是第一級需求

另外 `src/petals/__init__.py` 會在 import 時檢查 `transformers` 版本，表示這個 repo 對 API compatibility 很敏感。

## 7. 測試結構

`tests/` 底下的測試主題蠻完整，能反推出專案關心的穩定面：

- `test_full_model.py`: 完整模型流程
- `test_remote_sequential.py`: 遠端 sequential 執行
- `test_sequence_manager.py`: routing 邏輯
- `test_tensor_parallel.py`: tensor parallel
- `test_speculative_generation.py`: speculative decoding / generation
- `test_peft.py`: PEFT / adapter 能力
- `test_cache.py`: cache 行為
- `test_server_stats.py`: server 統計資訊

如果想從測試反推功能，這個資料夾很值得讀。

## 8. 其他資料夾的角色

- `examples/`: notebook 範例，偏教學與 prompt tuning
- `benchmarks/`: forward / inference / training benchmark
- `Dockerfile`: 容器化部署入口

其中 `run_dht.py` 特別值得注意，因為它代表 Petals 不只是「啟模型」而已，也考慮到建立 bootstrap node / private swarm 這種基礎設施需求。

## 9. 我對這個 repo 的理解摘要

這個 repo 的本質可以用一句話概括：

> Petals 用 Hugging Face 風格的 API，包裝一套基於 `hivemind` 的 P2P 分散式大模型執行系統。

它有三條主線：

- client：讓使用者像平常一樣載入與呼叫模型
- routing：找到哪些遠端節點能合作完成一條推論路徑
- server：讓節點持有部分模型並對外提供 block 計算服務

它不是單純推論 SDK，而是一個包含：

- network discovery
- block hosting
- distributed inference
- fine-tuning support
- deployment tooling

的完整系統。

## 10. 後續建議的深入方向

如果接下來要更深入讀這個 repo，我建議按這三條路徑選一條：

### 路徑 A：看使用者怎麼發出推論

聚焦檔案：

- `src/petals/utils/auto_config.py`
- `src/petals/client/from_pretrained.py`
- `src/petals/client/remote_generation.py`

適合想知道「使用者呼叫 generate 時，底層到底發生什麼事」的人。

### 路徑 B：看 server 怎麼對外提供 blocks

聚焦檔案：

- `src/petals/cli/run_server.py`
- `src/petals/server/server.py`
- `src/petals/server/backend.py`
- `src/petals/server/handler.py`

適合想知道「一台機器怎麼加入 swarm 並服務模型 layers」的人。

### 路徑 C：看 routing 怎麼找最佳路徑

聚焦檔案：

- `src/petals/client/routing/sequence_manager.py`
- `src/petals/client/routing/sequence_info.py`
- `src/petals/utils/dht.py`

適合想知道「Petals 怎麼做節點選擇與路由」的人。

## 11. 一句話總結

Petals 是一個把超大 LLM 拆成網路節點共同執行的分散式系統，而這個 repo 同時包含了 client API、routing、server runtime、模型適配層與部署工具，是一個完整的 distributed LLM runtime。