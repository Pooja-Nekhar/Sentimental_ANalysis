# Sentimental_ANalysis
# 🗳️ Cross-Platform Sentiment Analysis for Karnataka Election Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Transformers](https://img.shields.io/badge/HuggingFace-XLM--RoBERTa-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-77%25-green)

**MTech Project — 2024-25**  
**Student:** Pooja K V | **USN:** 24MTRCS007  
**Domain:** Computer Science & Engineering

---

## 📌 Project Overview

This project implements a **Cross-Platform Sentiment Analysis System** using Deep Learning to predict the likely winner of the upcoming Karnataka Legislative Assembly elections. The system collects political opinion data from 6 online platforms, processes multilingual text in English and Kannada, and uses a fine-tuned **XLM-RoBERTa** model to classify sentiment and generate party-wise predictions.

---

## 🏆 Prediction Result

| Party | Sentiment Score | Positive % | Rank |
|-------|----------------|------------|------|
| JDS   | +0.043 | 25.2% | 🥇 1st |
| Congress | -0.006 | 26.6% | 🥈 2nd |
| BJP   | -0.067 | 22.1% | 🥉 3rd |

**Predicted Winner: JDS (Janata Dal Secular)** 🏆

---

## 📊 System Architecture

```
Data Sources → Collection → Preprocessing → Sentiment Labeling → XLM-RoBERTa → Prediction
```

| Stage | Details |
|-------|---------|
| Data Collection | YouTube API, Google News RSS, The Hindu, NDTV, TOI, Wikipedia |
| Records | 1,218 collected → 1,063 after cleaning |
| Sentiment Labeling | VADER + Sentiment140 augmentation (3,063 records) |
| Model | XLM-RoBERTa (twitter-xlm-roberta-base-sentiment) |
| Accuracy | 77% validation accuracy |
| Dashboard | Streamlit + Plotly |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/karnataka-election-sentiment.git
cd karnataka-election-sentiment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
karnataka-election-sentiment/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── data_collection.py          # Collect data from all platforms
│   ├── preprocess.py               # Clean and label data
│   ├── train.py                    # Train XLM-RoBERTa model
│   └── predict.py                  # Run party-wise prediction
├── data/
│   ├── raw/                        # Raw collected data
│   │   ├── youtube/
│   │   ├── twitter/
│   │   ├── newspapers/
│   │   └── wikipedia/
│   └── processed/                  # Cleaned and labeled data
│       ├── master_dataset.csv
│       ├── cleaned_dataset.csv
│       ├── combined_dataset.csv
│       └── prediction_results.csv
└── model/                          # Saved XLM-RoBERTa model
    ├── config.json
    ├── model.safetensors
    └── tokenizer files
```

---

## 🔧 Full Pipeline

### Step 1 — Collect Data
```bash
# Set your YouTube API key
export YOUTUBE_API_KEY="your_key_here"
python src/data_collection.py
```

### Step 2 — Preprocess
```bash
python src/preprocess.py
```

### Step 3 — Train Model
```bash
python src/train.py
```

### Step 4 — Run Prediction
```bash
python src/predict.py
```

### Step 5 — Launch Dashboard
```bash
streamlit run app.py
```

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | cardiffnlp/twitter-xlm-roberta-base-sentiment |
| Task | 3-class sentiment classification |
| Languages | English + Kannada |
| Epochs | 5 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Optimizer | AdamW |
| Loss | Weighted CrossEntropyLoss |
| GPU | NVIDIA Tesla T4 |

### Accuracy Progression

| Round | Model | Accuracy |
|-------|-------|----------|
| Round 1 | mBERT + TextBlob labels | 64.79% |
| Round 2 | XLM-RoBERTa + VADER labels | 70.42% |
| Round 3 | XLM-RoBERTa + Combined dataset | **77.00%** |

---

## 📱 Dashboard Features

- 📊 **Dataset Overview** — Key metrics and system pipeline
- 🏆 **Election Prediction** — Winner announcement with charts
- 📈 **Sentiment Analysis** — Party-wise sentiment breakdown
- 🌐 **Platform Analysis** — Records by platform and heatmap
- 🤖 **Live Predictor** — Real-time sentiment prediction for any text
- 📰 **Fresh News** — Collect latest Karnataka political news

---

## 🛠️ Technologies Used

- **Python 3.12**
- **PyTorch 2.0** — Deep learning framework
- **HuggingFace Transformers** — XLM-RoBERTa model
- **Streamlit** — Interactive dashboard
- **Plotly** — Data visualizations
- **VADER** — Sentiment labeling
- **Google Colab T4 GPU** — Model training

---

## 📚 References

1. Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers
2. Conneau et al. (2020) — Unsupervised Cross-lingual Representation Learning at Scale
3. Hutto & Gilbert (2014) — VADER: A Parsimonious Rule-based Model for Sentiment Analysis
4. Tumasjan et al. (2010) — Predicting Elections with Twitter

---

## 👩‍💻 Author

**Pooja K V**  
MTech — Computer Science & Engineering  
USN: 24MTRCS007  
Deemed University | 2024-25

---

## 📄 License

This project is for academic purposes only.
