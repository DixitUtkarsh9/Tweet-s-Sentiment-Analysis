# Tweet's Sentiment Analysis

## Overview
This project focuses on sentiment analysis of tweets using state-of-the-art Natural Language Processing (NLP) models. The dataset contains approximately 32,000 tweets, and the best-performing model, BERT, achieved an accuracy of **99.64%**. A comparative analysis was also conducted using various transformer-based architectures.

## Features
- **BERT-based Sentiment Analysis**: Achieved high accuracy in classifying tweet sentiments.
- **Comparative Analysis of Transformers**: Evaluated LSTM, Bi-LSTM, XLNet, RoBERTa, ALBERT, and DistilBERT.
- **Scalable and Efficient**: Optimized preprocessing and training for large-scale datasets.

## Dataset
- Contains **~32,000 tweets** labeled with sentiment categories.
- Preprocessing steps include text cleaning, tokenization, and handling special characters.

## Technologies Used
- Python
- TensorFlow / PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook / Google Colab

## Installation
To set up the project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/Tweet-Sentiment-Analysis.git
cd Tweet-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage
To train the BERT model:

```bash
python train_bert.py
```

To perform sentiment analysis on new tweets:

```bash
python predict.py --text "This is a great day!"
```

## Results
| Model | Accuracy |
|--------|------------|
| BERT | **99.64%** |
| LSTM | Comparative Performance |
| Bi-LSTM | Comparative Performance |
| XLNet | Comparative Performance |
| RoBERTa | Comparative Performance |
| ALBERT | Comparative Performance |
| DistilBERT | Comparative Performance |

## Future Enhancements
- Fine-tune models with larger and more diverse datasets.
- Deploy as a web API for real-time sentiment analysis.
- Implement an interactive dashboard for sentiment visualization.

