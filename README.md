# NBME_Daberta_v3 【inference】

🏥 **A deep learning solution for clinical patient note scoring using DeBERTa v3 transformer model**

## 📋 Overview

This project implements a token-level binary classification system for the NBME (National Board of Medical Examiners) Score Clinical Patient Notes competition. The model identifies specific text spans in clinical patient notes that correspond to medical features, using Microsoft's DeBERTa v3 as the backbone architecture.

## 🎯 Problem Description

The task involves:
- **Input**: Clinical patient notes (text) and medical features
- **Output**: Character-level spans indicating where features are mentioned in the notes
- **Approach**: Token-level binary classification with span extraction post-processing

## 🏗️ Architecture

### Model Components

1. **Backbone**: DeBERTa v3 Base model for contextual understanding
2. **Classification Head**: Linear layer with dropout for token-level binary prediction
3. **Loss Function**: BCE with logits loss for training
4. **Post-processing**: Character-level probability mapping and threshold-based span extraction

### Key Features

- ✅ **Multi-fold ensemble**: 5-fold cross-validation averaging
- ✅ **Adaptive thresholding**: Automatic optimal threshold search
- ✅ **Memory efficient**: Gradient checkpointing and model cleanup
- ✅ **Robust preprocessing**: Handles various text formats and edge cases

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers pandas numpy tqdm
```

### Required Dependencies

```python
torch>=1.9.0
transformers>=4.21.0
pandas>=1.3.0
numpy>=1.21.0
tqdm>=4.62.0
```

### Dataset Structure

```
data/
├── test.csv              # Test set with IDs
├── features.csv          # Medical features descriptions
├── patient_notes.csv     # Clinical patient notes
└── models/
    ├── fold0.pt          # Trained model weights
    ├── fold1.pt
    ├── fold2.pt
    ├── fold3.pt
    └── fold4.pt
```

### Usage

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/nbme-deberta-v3-inference.git
cd nbme-deberta-v3-inference
```

2. **Prepare your data**:
   - Place test data in the expected directory structure
   - Ensure model checkpoints are available

3. **Run inference**:
```bash
python nbme-deberta-v3-inference.py
```

4. **Output**:
   - `submission.csv`: Final predictions ready for submission
   - `inference.log`: Detailed logging information

## ⚙️ Configuration

### CFG Class Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `debug` | `False` | Enable debug mode |
| `num_workers` | `0` | DataLoader workers |
| `model` | `deberta_base_cache` | Model path/name |
| `batch_size` | `8` | Inference batch size |
| `max_len` | `512` | Maximum sequence length |
| `seed` | `42` | Random seed |
| `n_fold` | `5` | Number of folds |

### Threshold Search

The system automatically searches for optimal prediction thresholds in the range `[0.45, 0.56]` with 0.01 step size.

## 📊 Model Performance

### Key Metrics
- **F1 Score**: Optimized through threshold tuning
- **Inference Speed**: ~8 samples per batch on CPU
- **Memory Usage**: Optimized with gradient accumulation

### Post-processing Pipeline

1. **Token to Character Mapping**: Convert token-level predictions to character-level probabilities
2. **Threshold Application**: Apply optimized threshold for binary decisions
3. **Span Extraction**: Identify continuous character spans above threshold
4. **Format Conversion**: Convert to competition submission format

## 🔧 Technical Details

### Model Architecture

```python
class DebertaForTokenBinary(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
```

### Data Processing

- **Tokenization**: DeBERTa tokenizer with special tokens
- **Truncation**: `only_second` strategy to preserve feature text
- **Padding**: Max length padding to 512 tokens
- **Attention**: Full attention mechanism across sequence

## 📁 File Structure
```
.
├── nbme_deberta_v3_inference.py    # Main inference script
├── models/
│   ├── fold0.pt                    # Model checkpoints
│   ├── fold1.pt
│   └── ...
├── data/
│   ├── test.csv                    # Test data
│   ├── features.csv                # Feature definitions
│   └── patient_notes.csv           # Patient notes
└── submission.csv                  # Output predictions
```

## 🔍 Error Handling

The system includes comprehensive error handling for:
- Missing model checkpoints
- Data loading failures
- Memory management issues
- Tokenization errors

## 📈 Optimization Features

1. **Memory Management**: Automatic garbage collection and CUDA cache clearing
2. **Ensemble Averaging**: Multi-fold prediction averaging for robustness
3. **Adaptive Thresholding**: Automatic optimal threshold discovery
4. **Efficient Batching**: Optimized batch processing for inference

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the DeBERTa model architecture
- NBME for providing the clinical dataset
- Hugging Face for the transformers library
- Kaggle community for insights and discussions
