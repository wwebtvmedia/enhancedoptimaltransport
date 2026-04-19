# Plan: Shared Embedding Network for Multimodal Schrödinger Bridge

## 🎯 Objective
Replace BPE-based text encoding with a neural network that creates a shared embedding space for both images and text, enabling "smart embedded" representations for the Schrödinger Bridge framework.

---

## 🏁 Implementation Status

| Component | Status | Notes |
| :--- | :---: | :--- |
| **Neural Tokenizer (CNN)** | 🚧 | Implementation in progress. |
| **Shared Embedding Design** | ✅ | Dimensions and mapping finalized. |
| **Contrastive Loss (InfoNCE)** | ✅ | Logic defined. |
| **Data Pipeline Integration** | 🚧 | STL10 descriptions mapped. |
| **Three-Phase Integration** | 📅 | Scheduled after VAE baseline. |

---

## 🏗️ Proposed Architecture

### 1. Neural Text Tokenizer (No BPE)
- **Input**: Raw UTF-8 bytes (0-255).
- **Structure**: 1D CNN with adaptive pooling.
- **Output**: Fixed-length embedding (512-dim).

### 2. Shared Embedding Space
- **Image Path**: VAE Encoder $\to$ Linear Projection (1152 $\to$ 512).
- **Text Path**: Neural Tokenizer $\to$ Fixed Embedding (512).
- **Alignment**: Symmetric InfoNCE loss (Contrastive Learning).

---

## 📅 Execution Roadmap

### Phase 1: Core Components 🚧
- [x] Define `NeuralTokenizer` class with 1D convolutions.
- [ ] Implement `ContrastiveLoss` (InfoNCE).
- [ ] Update `config.py` with `USE_NEURAL_TOKENIZER` flags.

### Phase 2: Data Pipeline 🚧
- [ ] Add raw text descriptions for STL10 classes in `data_management.py`.
- [ ] Implement byte-conversion utility for text inputs.
- [ ] Verify dimensional consistency across batch loaders.

### Phase 3: Training & Alignment 📅
- [ ] Implement Phase 1.5 (Contrastive Pre-alignment).
- [ ] Integrate Contrastive loss into the main `EnhancedLabelTrainer`.
- [ ] Monitor embedding separation via t-SNE during training.

---

## 📊 Success Criteria
1. **Retrieval Accuracy**: Recall@1 for Image $\to$ Text retrieval $> 80\%$.
2. **Stability**: No NaNs introduced by the high-magnitude contrastive gradients.
3. **Compatibility**: Existing label-based inference remains functional.

---
**Author:** Gemini CLI Agent  
**Last Updated:** April 19, 2026
