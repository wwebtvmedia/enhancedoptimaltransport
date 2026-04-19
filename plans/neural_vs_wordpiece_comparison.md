# Comparison: Neural Tokenizer vs. WordPiece

## 🎯 Objective
Evaluate the trade-offs between a custom end-to-end Neural Tokenizer and the industry-standard WordPiece algorithm for the Schrödinger Bridge multimodal task.

---

## ⚖️ Trade-off Matrix

| Feature | Neural Tokenizer (Byte-level) | WordPiece (BERT-style) |
| :--- | :--- | :--- |
| **Vocabulary** | ❌ None (uses 256 bytes) | ✅ Fixed (30k+ tokens) |
| **Out-of-Vocab** | ✅ Impossible (handles all bytes) | ⚠️ Uses `[UNK]` token |
| **Implementation** | ✅ Pure PyTorch (No deps) | ⚠️ Requires `tokenizers` lib |
| **Speed** | ⚠️ Moderate (Conv1D pass) | ✅ Fast (O(1) lookup) |
| **Adaptability** | ✅ Learns task-specific features | ❌ Pre-defined segmentation |

---

## 🔍 Detailed Analysis

### 1. Neural Tokenizer (Custom CNN)
- **Mechanism**: Processes raw UTF-8 bytes through a 1D Convolutional stack.
- **Pros**: Zero dependencies, fully differentiable, works on any language or symbol.
- **Cons**: Requires more training data to learn "concepts" from raw bytes.

### 2. WordPiece (Tokenization)
- **Mechanism**: Greedy subword merging based on likelihood.
- **Pros**: Highly efficient for common words, reduced sequence length.
- **Cons**: "Black-box" segmentation, library dependency, rigid vocabulary.

---

## 🏆 Recommendation
**Proceed with the Neural Tokenizer.**

**Rationale:**
1. **Self-Contained Implementation**: Aligns with the project's goal of being a lightweight, standalone research framework.
2. **"Smart Embedded" Goal**: By learning embeddings directly from bytes, the model can capture subtle character-level patterns (e.g., prefixes/suffixes) that WordPiece might fragment arbitrarily.
3. **Multimodal Harmony**: The CNN architecture of the tokenizer mirrors the VAE's spatial processing, creating a more cohesive architectural philosophy.

---

## 📅 Evaluation Plan
- [ ] Train baseline with label-only conditioning.
- [ ] Train experimental branch with Neural Tokenizer + Contrastive Loss.
- [ ] Compare Recall@1 metrics for STL10 class descriptions.

---
**Author:** Gemini CLI Agent  
**Last Updated:** April 19, 2026
