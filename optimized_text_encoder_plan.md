# Alternative Plan: Optimized BPE Text Encoder

> **Note**: This is an alternative proposal using Byte-Pair Encoding (BPE). The project is currently prioritizing the **Byte-level Neural Tokenizer** (CNN-based) for its zero-dependency and end-to-end differentiable nature.

---

## 🎯 Objective
Design a lightweight BPE-based text encoder that provides natural language understanding with a fixed vocabulary, optimized for low-resource environments.

---

## 🏁 Design Status: [Alternative]

| Component | Status | Notes |
| :--- | :---: | :--- |
| **BPE Tokenizer** | ✅ | Algorithm defined. |
| **Transformer Encoder** | ✅ | Linear-attention design finalized. |
| **CLIP Compatibility** | ⚠️ | High effort to align weights. |
| **Self-Supervised MLM** | 📅 | Proposed for pre-training. |

---

## 🏗️ Architecture: Transformer + BPE

### 1. BPE Tokenizer (`bpe_tokenizer.py`)
- **Vocab Size**: 10,000 tokens.
- **Base**: 256 bytes + learned merges.
- **Special Tokens**: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`.

### 2. Linear Transformer Encoder
- **Layers**: 4.
- **Heads**: 8.
- **Complexity**: $O(N)$ via Linear Attention (ELU-based kernels).
- **Projection**: Maps sequence context to a 512-dim bottleneck.

---

## ⚖️ Why use BPE over Bytes?
1. **Efficiency**: Longer concepts compressed into fewer tokens, reducing sequence length.
2. **Prior Knowledge**: Can be pre-trained on generic text corpora (Wiki/Books) more effectively than raw bytes.
3. **Common Practice**: Aligns with BERT/GPT tokenization standards.

---

## 📅 Proposed Execution (If Activated)
- [ ] Implement `BPETokenizer` with greedy longest-match.
- [ ] Implement `LinearAttention` blocks in `models.py`.
- [ ] Train on a subset of ImageNet captions to verify semantic clustering.

---
**Author:** Gemini CLI Agent  
**Last Updated:** April 19, 2026
