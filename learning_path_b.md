# 路徑 B：Server 怎麼對外提供 blocks

> 目標：理解一台機器如何加入 Petals swarm、持有部分模型 layers，並持續接受 client 的計算請求。

---

## 1. 整體架構一覽

啟動一個 Petals server 涉及四個主要角色，從外到內分別是：

```
CLI (run_server.py)
  └─ Server (server.py)          ← 全局 orchestrator，負責 rebalance 與生命週期
       └─ ModuleContainer         ← 一次 serving 週期（持有特定 block 集合）
            ├─ TransformerBackend (backend.py)  ← 每個 block 的計算核心
            └─ TransformerConnectionHandler (handler.py)  ← RPC 請求入口
```

`Server` 是一個不斷重啟 `ModuleContainer` 的 loop。每次重啟都可能換一批不同的 blocks，這就是 Petals 的自動 rebalance 機制。

---

## 2. run_server.py：CLI 入口

**檔案**：[src/petals/cli/run_server.py](src/petals/cli/run_server.py)

這個檔案只做兩件事：

1. **解析大量 CLI 參數**，包含：
   - 模型路徑（`--converted_model_name_or_path`）
   - 要服務哪幾個 blocks（`--num_blocks` 或 `--block_indices`）
   - 網路設定（`--port`、`--public_ip`、`--initial_peers`）
   - 硬體設定（`--device`、`--torch_dtype`、`--quant_type`、`--tensor_parallel_devices`）
   - 效能設定（`--max_batch_size`、`--attn_cache_tokens`、`--throughput`）

2. **建立 Server 物件並呼叫 `server.run()`**：

```python
server = Server(**args, host_maddrs=host_maddrs, ...)
try:
    server.run()
except KeyboardInterrupt:
    ...
finally:
    server.shutdown()
```

值得注意的是 `--throughput` 參數，設為 `"auto"` 時，啟動時會自動評估這台機器的計算與網路吞吐量，結果會被緩存並上報給 DHT，讓 client 端的 routing 能做出更好的路徑選擇。

---

## 3. server.py：兩個核心類別

**檔案**：[src/petals/server/server.py](src/petals/server/server.py)

### 3.1 `Server`：全局 orchestrator

`Server.__init__()` 在啟動時完成以下準備工作：

| 任務 | 說明 |
|------|------|
| 載入 block config | `AutoDistributedConfig.from_pretrained()`，讀模型 metadata |
| 決定 DHT prefix | 用來在 DHT 上識別這個模型的所有 blocks |
| 設定裝置與 dtype | 自動偵測 CUDA/MPS/CPU，解析量化類型 |
| 計算 attention cache 大小 | 根據 `hidden_size`、`attn_cache_tokens` 估算每個 block 需要多少 cache 記憶體 |
| 估算可服務的 block 數 | `_choose_num_blocks()` 根據 GPU 記憶體自動算出能放幾個 block |
| 建立 DHT 連線 | 加入 swarm，決定是否需要透過 relay 存取 |
| 評估吞吐量 | `get_server_throughput()` 測速並緩存結果 |

`Server.run()` 是一個無限 loop：

```python
def run(self):
    while True:
        block_indices = self._choose_blocks()          # 問 DHT 哪些 blocks 最缺
        self.module_container = ModuleContainer.create(...)  # 載入模型並啟動
        self.module_container.ready.wait()             # 等待就緒

        while True:
            if self.stop.wait(timeout):
                return
            if not self.module_container.is_healthy(): # 子程序掛掉就重啟
                break
            if self._should_choose_other_blocks():     # Swarm 失衡就換 blocks
                break
        self.module_container.shutdown()
        self._clean_memory_and_fds()                   # 清 GPU 記憶體後重來
```

**rebalance 機制**：`_should_choose_other_blocks()` 會定期查 DHT，如果當前 swarm 的吞吐量低於最優解的 75%（`balance_quality` 預設值），就會觸發換 blocks。這讓整個 swarm 能自動向最佳分布演化。

### 3.2 `ModuleContainer`：一次 serving 週期

`ModuleContainer.create()` 是這個類別的工廠方法，負責真正的初始化工作：

```
1. 建立 MemoryCache（管理 KV cache 的 GPU 記憶體）
2. 啟動 ModuleAnnouncerThread（向 DHT 宣告「我正在加入」）
3. 對每個 block_index：
   a. load_pretrained_block() → 從磁碟/HuggingFace 載入 block 權重
   b. convert_block() → 量化、tensor parallel 切分、凍結梯度
   c. 包裝成 TransformerBackend（含三個 task pool）
4. merge_inference_pools_inplace() → 合併所有 block 的 inference pool
5. 建立 num_handlers 個 TransformerConnectionHandler
6. 建立 RuntimeWithDeduplicatedPools 開始處理批次
7. 向 DHT 宣告「ONLINE」
```

**`ModuleAnnouncerThread`**：這是一個背景 thread，每隔 `update_period` 秒就呼叫一次 `declare_active_modules()`，把自己持有的 blocks 寫入 DHT，包含：
- 持有的 block 範圍（`start_block`、`end_block`）
- 當前狀態（`JOINING` / `ONLINE` / `OFFLINE`）
- 剩餘 cache 空間（client routing 會用這個判斷是否還能接新 session）
- 對後繼 server 的 ping 延遲（幫助 client 選擇最低延遲路徑）

---

## 4. backend.py：TransformerBackend

**檔案**：[src/petals/server/backend.py](src/petals/server/backend.py)

每個 block 被包裝成一個 `TransformerBackend`，它繼承自 `hivemind.ModuleBackend`，核心職責是提供三種計算能力，各自對應一個 `PrioritizedTaskPool`：

| Pool | 用途 |
|------|------|
| `inference_pool` | 自回歸推論（帶 KV cache） |
| `forward_pool` | 完整 forward pass（不帶 cache，用於 fine-tuning） |
| `backward_pool` | 反向傳播（計算梯度，用於 fine-tuning） |

### 4.1 inference_step：推論的核心

```python
@torch.inference_mode()
def inference_step(self, hidden_states, hypo_ids, inference_info):
    with self.memory_cache.use_cache(*inference_info.cache_handles) as cache_tensors:
        self._reorder_cache_inplace(cache_tensors, hypo_ids)   # beam search 支援
        layer_past = self._select_layer_past(cache_tensors, inference_info.prefix_length)
        for offset in range(0, seq_len, max_chunk_length):     # 分 chunk 避免 OOM
            output_chunk, new_kvs = self.module.forward(hidden_states_chunk, layer_past=layer_past, use_cache=True)
            layer_past = new_kvs
        self._update_cache_inplace(cache_tensors, new_kvs, inference_info.prefix_length)
        return (output_hidden_states,)
```

幾個設計細節：
- **chunked processing**：`max_chunk_size_bytes` 控制一次處理多少 token，避免長序列的 attention matrix 撐爆 GPU。
- **KV cache 存在 MemoryCache 裡**：cache 是跨 step 持久存在的，每個 inference session 有自己的 cache handle。
- **`hypo_ids`**：beam search 時用來重排 cache 的索引，確保不同假設（hypothesis）的 cache 不會混在一起。

### 4.2 merged inference pool

`merge_inference_pools_inplace()` 把所有 block 的 inference pool 合併成一個 `_MergedInferenceStep`：

```python
class _MergedInferenceStep:
    def __call__(self, hidden_states, hypo_ids, inference_infos, *optional_prompts):
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, :optional_prompt.shape[1]] += optional_prompt  # prompt tuning
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        return (hidden_states,)
```

這意味著一個 handler 的 inference 請求可以在一次 GPU kernel call 中跑過這台機器上的所有 blocks，減少 kernel launch 的 overhead。`optional_prompts` 則是 soft prompt tuning 的支援點。

---

## 5. handler.py：TransformerConnectionHandler

**檔案**：[src/petals/server/handler.py](src/petals/server/handler.py)

`TransformerConnectionHandler` 繼承自 `hivemind.ConnectionHandler`，是 server 對外的 RPC 介面。它是 async 的（asyncio），處理來自 client 的請求。

### 5.1 三種主要 RPC

| RPC | 說明 |
|-----|------|
| `rpc_inference` | 自回歸推論，是一個 streaming async generator |
| `rpc_forward` / `rpc_forward_stream` | 完整 forward（fine-tuning 用） |
| `rpc_backward` / `rpc_backward_stream` | 反向傳播（fine-tuning 用） |

### 5.2 rpc_inference：推論的流程

```
1. 從 metadata 讀取 max_length、session_id、alloc_timeout 等資訊
2. 呼叫 _allocate_cache() 從 MemoryCache 申請 KV cache 空間
3. 進入 iterate_rpc_inference() loop：
   - 每個 step 從 client 收到新的 hidden_states
   - 交給 merged inference pool 執行（非同步，等待 GPU 結果）
   - yield 結果回給 client
   - 若 metadata 指定了 next_servers，非同步 push 輸出給下一個 server
```

**session 管理**：多個 handler process 之間透過 `handler_event_queues`（multiprocessing Queue）共享 session 資訊。當一個 session 的後續請求（`rpc_push`）被路由到不同的 handler 時，會透過這個 queue 轉發。這解決了 async 多 handler 環境下的 session 親和性問題。

**server-to-server push**（`_push_outputs`）：推論結果除了回傳給 client，也可以直接 push 給下一個 server，減少一來一回的 RTT。這在 `can_push=True` 時觸發，metadata 裡的 `next_servers` 欄位告訴當前 server 要把輸出送到哪裡。

### 5.3 cache 分配（`_allocate_cache`）

```python
async with self._allocate_cache(backends, batch_size=batch_size, max_length=max_length, timeout=alloc_timeout) as cache_handles:
    ...
```

KV cache 是有限資源。如果記憶體不夠，server 會等待最多 `alloc_timeout` 秒（預設 600 秒），直到有 session 結束釋放空間。這個超時時間是 client 在發出請求時帶過來的，server 以此決定要等多久。

---

## 6. 完整啟動流程總結

```
python -m petals.cli.run_server meta-llama/Meta-Llama-3.1-70B
  │
  ├─ 解析 CLI 參數
  ├─ Server.__init__()
  │    ├─ 載入 block config
  │    ├─ 建立 DHT 連線（加入 swarm）
  │    ├─ 估算可持有的 block 數
  │    └─ 評估並緩存吞吐量
  │
  └─ Server.run()  [無限 loop]
       ├─ _choose_blocks()：查 DHT 找最需要的 block 範圍
       ├─ ModuleContainer.create()
       │    ├─ 建立 MemoryCache
       │    ├─ 啟動 ModuleAnnouncerThread（宣告 JOINING）
       │    ├─ 對每個 block：load → quantize → wrap as TransformerBackend
       │    ├─ merge_inference_pools_inplace()
       │    ├─ 建立 N 個 TransformerConnectionHandler
       │    ├─ 建立 RuntimeWithDeduplicatedPools
       │    └─ 宣告 ONLINE
       │
       ├─ [持續對外服務，接受 rpc_inference / rpc_forward / rpc_backward]
       │
       ├─ [每隔 ~120 秒檢查 swarm 是否失衡]
       │    └─ 若失衡 → shutdown ModuleContainer → 換 blocks → 重啟
       │
       └─ [子程序掛掉] → 重啟
```

---

## 7. 與其他路徑的接口

- **路徑 A（client）** 透過 DHT 找到這台 server 持有的 blocks，然後呼叫 `rpc_inference` 等 RPC。
- **路徑 C（routing）** 讀取 server 上報到 DHT 的 `next_pings` 等資訊，決定是否把這台 server 納入最佳路徑。
- `TransformerConnectionHandler._push_outputs()` 實現了 server 主動 push 給下一個 server 的機制，縮短了 client 不需要「中轉」的推論延遲。

---

## 9. 補充：Block 是什麼？

在 Petals 的語境中，**block** 指的是大型語言模型的一個 **Transformer Block（Transformer 層）**。

### 模型結構

一個大型語言模型（例如 LLaMA 405B）是由幾百個 Transformer blocks 堆疊而成的：

```
Embedding
  └─ Block 0   (Attention + FFN)
  └─ Block 1   (Attention + FFN)
  └─ ...
  └─ Block N   (Attention + FFN)
LM Head
```

每個 block 包含一個 Multi-Head Attention 層和一個 Feed-Forward Network（FFN）層，是模型計算的基本單位。

### Block 在 Petals 中的角色

單一機器的 GPU 記憶體無法放下整個模型，Petals 的解法是把 blocks **分散到多台機器**：

- 機器 A 持有 block 0–20
- 機器 B 持有 block 21–50
- 機器 C 持有 block 51–80

推論時，hidden states 沿 block 0 → block N 的順序在各台機器間依序傳遞，每台只算自己那幾層，最後拼湊出完整結果。

### Block 的生命週期（server 視角）

```
load_pretrained_block()    ← 從磁碟 / HuggingFace 載入權重
      ↓
convert_block()            ← 量化（INT8/NF4）、tensor parallel 切分、凍結梯度
      ↓
TransformerBackend         ← 包裝成可接受 RPC 請求的計算單元
      ├─ inference_pool    ← 自回歸推論（帶 KV cache）
      ├─ forward_pool      ← 完整 forward（fine-tuning）
      └─ backward_pool     ← 反向傳播（fine-tuning）
```

### 持有哪些 blocks？

由 `_choose_blocks()` 查詢 DHT 決定，優先補足 swarm 中覆蓋不足的 block 範圍。Server 決定持有哪些 blocks 後，會透過 `ModuleAnnouncerThread` 將 block 範圍寫入 DHT，讓 client 的 routing 知道如何找到這台機器。

簡單一句話：**block = 模型的一個 Transformer layer，是 Petals 分散式推論的最小分配單位**。

---

## 8. 補充：Petals Swarm 是什麼？

**Swarm** 就是一群透過 P2P 網路連在一起、共同「拼湊」出一個完整大模型的機器集合。

### 核心概念

一個超大模型（例如 LLaMA 405B）有幾百個 Transformer blocks。沒有任何一台機器能單獨放下整個模型，但：

- 機器 A 持有 block 0–20
- 機器 B 持有 block 21–50
- 機器 C 持有 block 51–80
- …

這些機器加在一起，就覆蓋了整個模型。這個集合就是 swarm。

### Swarm 如何運作

1. 每台 server 啟動後，透過 **DHT**（分散式雜湊表，由 `hivemind` 提供）宣告自己持有哪些 blocks
2. Client 查詢 DHT，找出能覆蓋完整模型的一條 server 路徑
3. Inference 時，hidden states 沿這條路徑逐站傳遞，每台機器只算自己那幾層

### Public swarm vs Private swarm

從 `run_server.py` 的參數可以看出兩種模式：

- **Public swarm**：連到 `PUBLIC_INITIAL_PEERS`（Petals 官方的 bootstrap nodes），加入公開的共享 swarm，任何人都可以用
- **Private swarm**：`--new_swarm` 或自訂 `--initial_peers`，自己建一個封閉的 swarm，只有自己的機器參與

### Swarm 會自動 rebalance

`Server.run()` 裡有一個 loop，每隔約 120 秒檢查一次整個 swarm 的 block 分布。如果某些 blocks 覆蓋不足導致吞吐量低於最優的 75%，server 會自動換掉自己持有的 blocks，往缺乏的地方補——這就是 `_should_choose_other_blocks()` 的作用。

簡單說：**swarm = 多台機器共同組成一個「虛擬完整模型」的協作網路**。
