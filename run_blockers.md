# Petals 啟動困難點整理

## 1. 環境完全未安裝

`petals` conda 環境是空的，只有 pip/setuptools，所有依賴都需要從頭安裝。

```bash
conda activate petals
pip install -e .  # 會踩到下面幾個衝突
```

---

## 2. 依賴版本衝突

### `huggingface-hub<1.0.0`
- `setup.cfg` 限制只能裝 0.x，最高到 0.36.2
- 目前 PyPI 最新是 1.21.0，pip 解析時直接報衝突
- **修法**：移除 `setup.cfg` 裡的 `<1.0.0` 上限，並確認 API 相容性

### `bitsandbytes==0.41.1`
- 鎖死在 2023 年的舊版
- 完全不支援 RTX 5060（Blackwell 架構）
- 最新版 0.49.2 才有 Blackwell 支援
- **修法**：改為 `bitsandbytes>=0.43.0`，並測試量化功能是否正常

### `transformers==4.43.1`
- 版本鎖死，且 `src/petals/__init__.py` 有 runtime assert：
  ```python
  assert version.parse("4.43.1") <= version.parse(transformers.__version__) < version.parse("4.44.0")
  ```
  裝錯版本時 import 直接崩潰
- **修法**：鎖死安裝 4.43.1，或同步修改 `setup.cfg` 與 `__init__.py` 的版本範圍

---

## 3. GPU 架構相容性（RTX 5060 / Blackwell）

- RTX 5060 是 2025 年的 Blackwell 架構（compute capability 12.x）
- 2023–2024 年的 ML 套件多數沒有預編譯 Blackwell 的 CUDA kernel
- 安裝後執行時可能出現 `CUDA error` 或 fallback 到 CPU
- CUDA 版本：13.0（Driver 581.34）
- 受影響套件：`bitsandbytes`、`hivemind`、`tensor_parallel` 等

---

## 4. Public Swarm 已停用

- `health.petals.dev` 的公共節點已無維運
- 即使本地安裝成功，`AutoDistributedModelForCausalLM` 連線時會 hang 或 timeout
- **替代方案**：
  - 自建 Private Swarm（需多台 GPU 機器，各自跑 `petals.cli.run_server`）
  - 改用 Ollama 在本機跑小模型（RTX 5060 8GB 支援到 ~13B 4-bit 量化）
  - 使用雲端推論 API（Groq、Together AI、Anthropic 等）

---

## 總結

| 困難點 | 嚴重程度 | 能否修 |
|--------|----------|--------|
| 環境未安裝 | 低 | 跑 `pip install -e .` 即可（但會踩到下面幾個） |
| `huggingface-hub<1.0.0` 衝突 | 高 | 需改 `setup.cfg` 放寬限制 |
| `bitsandbytes==0.41.1` 不支援 Blackwell | 高 | 需升級版本，確認 API 相容性 |
| `transformers` 版本 assert | 中 | 鎖死 4.43.1 或同步修改 `setup.cfg` + `__init__.py` |
| Blackwell GPU 整體相容性 | 中 | 逐套件測試才知道 |
| Public Swarm 不存在 | 高 | 需自建 swarm 或改用其他方案 |

> 根本原因：此 repo 最後更新為 2024 年中，依賴版本未考慮 2025–2026 年的環境，加上 public swarm 停用，需相當多修改才能跑起來。
