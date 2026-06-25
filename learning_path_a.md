# 路徑 A：使用者怎麼發出推論

> 聚焦問題：使用者呼叫 `generate` 時，底層到底發生什麼事？

## 涉及檔案

- `src/petals/utils/auto_config.py`
- `src/petals/client/from_pretrained.py`
- `src/petals/client/remote_generation.py`

---

## 第一站：`auto_config.py` — 分派正確的 class

**問題**：使用者只給了一個 model 名稱，Petals 怎麼知道要建立哪個 class？

```python
model = AutoDistributedModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-405B-Instruct"
)
```

流程拆解：

1. `AutoDistributedModelForCausalLM` 繼承自 `_AutoDistributedBase`，`_mapping_field = "model_for_causal_lm"`
2. `from_pretrained` 先呼叫 HuggingFace 的 `AutoConfig.from_pretrained()`，得知 `model_type = "llama"`
3. 查內部的 `_CLASS_MAPPING["llama"]`，取出 `.model_for_causal_lm`
4. 這個 mapping 是在 `petals/models/llama/__init__.py` 被 import 時、透過 `register_model_classes()` 填進去的
5. 結果取得 `DistributedLlamaForCausalLM`，然後呼叫它的 `.from_pretrained()`

**關鍵點**：`_CLASS_MAPPING` 是一個全域 dict，各個 model 子套件（llama、bloom、falcon…）在 import 時各自 register 自己。這是一個簡單的 plugin registry 模式。

---

## 第二站：`from_pretrained.py` — 只載入本地需要的部分

`FromPretrainedMixin.from_pretrained` 做了兩件重要的事：

### 版本相容性處理

```python
model_name_or_path = get_compatible_model_repo(model_name_or_path)
```

把 model 名稱轉換成 Petals 相容的版本。

### 跳過不需要的 checkpoint shards

```python
with ignore_keys(cls._keys_to_ignore_on_load_unexpected):
    return super().from_pretrained(...)
```

這裡有個關鍵機制：`patched_get_checkpoint_shard_files()` 直接 monkey-patch 了 HuggingFace 的下載函式。

**為什麼需要這樣做？**

405B 模型的 transformer blocks（大部分的權重）是放在遠端 server 的，client 只需要：

- embedding layer（把 token ID 轉成 hidden state）
- LM head（把最後的 hidden state 轉成 logits）

所以 `_keys_to_ignore_on_load_unexpected` 裡列的就是那些 transformer block 的 weight name pattern，讓 Petals 在讀 checkpoint index 時，直接把那些 shards 從 download list 中移除，**避免把幾百 GB 的 block 權重抓到本機**。

---

## 第三站：`remote_generation.py` — 管理 session 與 token 生成

這是路徑 A 最複雜的部分。`RemoteGenerationMixin.generate()` 包在 HuggingFace `GenerationMixin.generate()` 的外面，做了幾件額外的事：

### 3.1 Session 管理

```python
if session is not None:          # 明確指定 session
elif self.active_session:        # 已有 active session
else:                            # 建立新的 InferenceSession
```

`InferenceSession` 的用途是讓遠端 server 保存 **attention KV cache**。沒有 session 就代表每個 token 都要重跑全部的 prefix，效率很差。

新建 session 時，需要計算 `session_max_length`，這個長度決定遠端 server 要預留多少 cache 空間：

```python
session_max_length += max_length           # 或
session_max_length += inputs.shape[1] + max_new_tokens
```

### 3.2 `RemotePastKeyValues` — 假裝有 KV cache

HuggingFace 的 generation loop 預期 model 要回傳 `past_key_values` 做 KV cache 加速。

但 Petals 的 KV cache 是存在遠端 server 上的，client 端沒有。

解法：建一個假的 `RemotePastKeyValues` 物件，它繼承 `Cache` 但只追蹤「已看過幾個 token (`_seen_tokens`)」，每次被 HuggingFace 框架問到就回傳 `DUMMY`，不實際儲存任何 tensor。

### 3.3 Token 跳過機制

`_SkipTokensMixin.prepare_inputs_for_generation` 會把 `input_ids` 從 `_skipped_tokens` 的位置切掉：

```python
input_ids = input_ids[:, _skipped_tokens.get():]
```

用途是：當一個 session 被多次 `generate()` 呼叫時（例如 chatbot 的多輪對話），前一輪的 token 已經在遠端 server 的 cache 裡了，不需要重新 forward，只需要讓 HuggingFace 框架的統計層（例如 repetition_penalty）還能看到它們就好。

### 3.4 實際的 forward 路徑

最終 `super().generate()` 啟動 HuggingFace 的 generation loop，每個 step 的 forward 路徑是：

```
input_ids
→ 本地 embedding layer（client）
→ RemoteSequential（把 hidden states 傳到遠端 server，server 執行 transformer blocks）
→ 本地 LM head（client）
→ logits → 取樣出下一個 token
```

遠端部分的執行是透過 `RemoteSequential`，它會呼叫 `InferenceSession`，再透過 P2P 網路傳遞 activations。

---

## 整體流程圖

```
使用者呼叫 AutoDistributedModelForCausalLM.from_pretrained(model_name)
    │
    ├─ auto_config.py
    │     HF AutoConfig 讀 model_type
    │     查 _CLASS_MAPPING → DistributedLlamaForCausalLM
    │
    ├─ from_pretrained.py
    │     跳過 transformer block shards
    │     只下載 embedding + LM head 權重到本地
    │
使用者呼叫 model.generate(inputs, max_new_tokens=100)
    │
    └─ remote_generation.py
          建立 InferenceSession（遠端預留 attention cache）
          建立假的 RemotePastKeyValues
          → 進入 HF generation loop
               每步：embedding (local) → transformer blocks (remote) → LM head (local)
               遠端 server 保存 KV cache，只傳遞 hidden states
          回傳生成的 token sequences
```

---

## 關鍵洞察整理

| 問題 | 答案 |
|------|------|
| 為什麼 client 不會把整個模型下載下來？ | `from_pretrained.py` 用 monkey-patch 跳過 block shards |
| 為什麼推論比一般模型複雜？ | 每個 forward pass 的中間層計算需要透過 P2P 傳送 activations |
| 為什麼需要 session？ | 讓遠端 server 保存 KV cache，避免每個 token 都重跑 prefix |
| `RemotePastKeyValues` 為什麼是假的？ | 真正的 KV cache 在遠端，client 只需讓 HF 框架不報錯 |

---

## 下一步可以深入的方向

- **`inference_session.py`** — session 如何與遠端 server 建立連線、傳遞 activations
- **`remote_sequential.py`** — `RemoteSequential` 怎麼把 `forward()` 路由到遠端節點
- **routing 層**（路徑 C 的起點）— session 怎麼知道要連到哪些 server
