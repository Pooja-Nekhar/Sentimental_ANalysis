"""
Preprocessing Script
Cleans text, assigns party labels, labels sentiment using VADER
Run: python src/preprocess.py
"""

import os
import re
import pandas as pd
from datetime import datetime

OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\u0C80-\u0CFF.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_party(text):
    if not isinstance(text, str):
        return "general"
    text_lower        = text.lower()
    bjp_keywords      = ["bjp","bharatiya janata","yediyurappa","bommai","modi","amit shah","basavaraj","ಬಿಜೆಪಿ"]
    congress_keywords = ["congress","inc","siddaramaiah","dk shivakumar","rahul gandhi","kharge","sonia","ಕಾಂಗ್ರೆಸ್"]
    jds_keywords      = ["jds","janata dal","kumaraswamy","hd deve gowda","secular","ಜೆಡಿಎಸ್"]
    bjp_count         = sum(1 for k in bjp_keywords if k in text_lower)
    congress_count    = sum(1 for k in congress_keywords if k in text_lower)
    jds_count         = sum(1 for k in jds_keywords if k in text_lower)
    max_count         = max(bjp_count, congress_count, jds_count)
    if max_count == 0:                return "general"
    elif bjp_count == max_count:      return "BJP"
    elif congress_count == max_count: return "Congress"
    else:                             return "JDS"


def get_vader_sentiment(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    try:
        score = analyzer.polarity_scores(str(text))["compound"]
        if score >= 0.05:    return "positive"
        elif score <= -0.05: return "negative"
        else:                return "neutral"
    except Exception:
        return "neutral"


def preprocess(input_path="data/processed/master_dataset.csv"):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded: {len(df)} records")

    df["cleaned_text"] = df["text"].apply(clean_text)
    df["party"]        = df["cleaned_text"].apply(assign_party)
    df = df[df["cleaned_text"].str.len() > 20]
    df = df[df["cleaned_text"].notna()]

    output_path = f"{OUTPUT_DIR}/cleaned_dataset.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Cleaned dataset saved: {len(df)} records")
    print(df["party"].value_counts().to_string())

    # VADER labeling
    print("\nApplying VADER sentiment labels...")
    df["sentiment"] = df["cleaned_text"].apply(get_vader_sentiment)
    print(df["sentiment"].value_counts().to_string())

    labeled_path = f"{OUTPUT_DIR}/labelled_dataset.csv"
    df.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Labelled dataset saved: {labeled_path}")
    return df


if __name__ == "__main__":
    preprocess()
