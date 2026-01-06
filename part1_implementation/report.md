# Sentiment Analysis with Custom Transformer: Project Report

## 1. Introduction

This project implements a custom transformer-based sentiment classifier trained on the SST-2 (Stanford Sentiment Treebank) dataset. The goal is to classify movie review sentences as positive or negative while demonstrating understanding of transformer architecture, training best practices, and model interpretability through attention visualization.

---

## 2. Approach

### 2.1 Model Architecture

We implemented a **custom transformer encoder** from scratch rather than fine-tuning a pre-trained model. The architecture consists of:

| Component | Configuration |
|-----------|---------------|
| Token Embeddings | 30,522 vocab × 256 dim |
| Positional Embeddings | 128 positions × 256 dim (learnable) |
| Transformer Layers | 4 encoder blocks |
| Attention Heads | 8 heads per layer |
| Feed-Forward Dim | 512 (with GELU activation) |
| Classification Head | Linear(256 → 256) → Tanh → Linear(256 → 2) |
| Dropout | 0.1 throughout |

The model uses the **[CLS] token representation** (first token) for classification, following the BERT paradigm. Attention weights from all layers are captured for interpretability analysis.

### 2.2 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for transformers; decoupled weight decay |
| Learning Rate | 1e-4 | Conservative for training from scratch |
| LR Schedule | Cosine Annealing | Smooth decay with warm restarts |
| Batch Size | 32 | Balance between speed and gradient stability |
| Label Smoothing | 0.1 | Prevents overconfident predictions |
| Gradient Clipping | 1.0 | Stabilizes training |
| Early Stopping | Patience=3 | Prevents overfitting |

### 2.3 Dataset

**SST-2** from the GLUE benchmark:
- **Training set**: 67,349 sentences
- **Validation set**: 872 sentences
- **Task**: Binary sentiment classification (positive/negative)
- **Tokenizer**: BERT WordPiece (bert-base-uncased)

---

## 3. Results

### 3.1 Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|-------|------------|-----------|----------|---------|--------|
| 1 | 0.4787 | 81.37% | 0.5730 | 79.36% | 81.40% |
| 2 | 0.3588 | 90.71% | 0.5807 | 80.96% | 80.42% |
| 3 | 0.3191 | 93.14% | 0.5799 | 81.42% | 80.71% |
| 4 | 0.2921 | 94.79% | 0.6076 | 79.36% | 80.56% |

**Best Model**: Epoch 1 (Val Loss: 0.5730, Val Acc: 79.36%)

Training was terminated at epoch 4 due to early stopping (no improvement in validation loss for 3 consecutive epochs).

### 3.2 Classification Performance (Best Model)

```
              Precision    Recall    F1-Score   Support
─────────────────────────────────────────────────────────
Negative        0.8563     0.6963     0.7680      428
Positive        0.7519     0.8874     0.8140      444
─────────────────────────────────────────────────────────
Accuracy                              0.7936      872
Macro Avg       0.8041     0.7918     0.7910      872
```

### 3.3 Key Observations

1. **Overfitting**: Training accuracy reached 94.79% while validation accuracy plateaued at ~80%, indicating the model memorized training data rather than learning generalizable patterns.

2. **Class Imbalance in Predictions**: The model shows higher recall for positive class (88.74%) but lower precision (75.19%), suggesting a bias toward predicting positive sentiment.

3. **Early Convergence**: The best validation loss occurred at epoch 1, with subsequent epochs showing increasing validation loss despite decreasing training loss—a classic overfitting signature.

4. **Reasonable Baseline**: Achieving ~80% accuracy with a custom transformer trained from scratch (vs. 93%+ with fine-tuned BERT) demonstrates the architecture works but lacks the benefits of pre-training.

---

## 4. Analysis

### 4.1 Why Overfitting Occurred

| Factor | Impact |
|--------|--------|
| **No pre-training** | Model must learn language representations from only 67K examples |
| **Model capacity** | 4 layers × 8 heads may be too expressive for dataset size |
| **Limited regularization** | Only dropout (0.1) applied |

### 4.2 Attention Visualization Insights

The attention visualization module revealed:
- **Early layers** focus on local token relationships and punctuation
- **Later layers** develop broader attention patterns across the sentence
- **[CLS] token** increasingly attends to sentiment-bearing words (e.g., "great", "terrible") in deeper layers

### 4.3 Error Patterns

Based on the precision/recall imbalance:
- **False Positives** (predicting positive when negative): Lower precision on positive class suggests the model over-predicts positive sentiment
- **False Negatives on Negative Class**: 30% of negative reviews misclassified, likely due to subtle negativity or sarcasm

---

## 5. Potential Improvements

### 5.1 Addressing Overfitting

| Technique | Expected Impact |
|-----------|-----------------|
| **Pre-trained initialization** | Use BERT/DistilBERT weights instead of random init |
| **Data augmentation** | Back-translation, synonym replacement |
| **Increased dropout** | Try 0.2-0.3 in attention and feed-forward layers |
| **Reduce model size** | 2 layers instead of 4; smaller embed_dim |
| **Weight decay increase** | Try 0.05-0.1 instead of 0.01 |

### 5.2 Architecture Enhancements

- **Pre-layer normalization**: More stable training (used in GPT-2, modern transformers)
- **Relative positional encoding**: Better generalization to different sequence lengths
- **Pooling strategies**: Mean pooling or attention pooling instead of [CLS] only

### 5.3 Training Improvements

- **Learning rate warmup**: Gradual increase over first 10% of training
- **Longer training with lower LR**: 1e-5 to 5e-5 range with more epochs
- **Mixup/CutMix**: Data augmentation at embedding level

---

## 6. Conclusion

This project successfully implemented a custom transformer for sentiment analysis, achieving **79.36% validation accuracy** on SST-2. While this falls short of state-of-the-art results (~93% with fine-tuned BERT), the implementation demonstrates:

1. Correct transformer architecture with positional embeddings
2. Proper training infrastructure (checkpointing, early stopping, logging)
3. Attention visualization for model interpretability
4. Comprehensive evaluation with multiple metrics

The primary limitation is **overfitting due to training from scratch** on a relatively small dataset. The most impactful improvement would be initializing from pre-trained weights, which would likely boost accuracy by 10-15 percentage points.

---

## Appendix: Project Structure

```
sentiment_analysis/
├── model.py              # Transformer architecture
├── train.py              # Training loop with logging
├── evaluate.py           # Evaluation metrics
├── visualize_attention.py # Attention heatmaps
├── dataset.py            # SST-2 data loading
├── utils.py              # Utilities (checkpointing, seeding)
├── config.yaml           # Hyperparameters
├── checkpoints/          # Saved models
├── runs/                 # TensorBoard logs
└── attention_plots/      # Attention visualizations
```
### Running the Project

**1. Configure settings** (optional):

Edit `config.yaml` to adjust hyperparameters:

```yaml
device: "cuda"          # Change to "cpu" if no GPU
dataset:
  batch_size: 32        # Reduce to 16 if running out of memory
  num_workers: 0        # Keep 0 on Windows
training:
  epochs: 10
  lr: 1e-4
```

**2. Train the model**:

```bash
python train.py
```

This will:
- Download SST-2 dataset automatically (first run only)
- Train for up to 10 epochs with early stopping
- Save checkpoints to `checkpoints/`
- Generate attention visualizations in `attention_plots/`
- Log metrics to TensorBoard in `runs/`

**3. Monitor training with TensorBoard**:

```bash
python -m tensorboard.main --logdir=runs
```

Then open http://localhost:6006 in your browser.

**4. View attention visualizations**:

After training, check the `attention_plots/` directory for heatmaps showing which tokens the model attends to.
