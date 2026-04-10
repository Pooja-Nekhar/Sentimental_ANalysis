import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import re
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Karnataka Election Sentiment Analysis",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: bold;
    text-align: center;
    color: #1A237E;
    margin-bottom: 0.3rem;
}
.sub-title {
    font-size: 1.0rem;
    text-align: center;
    color: #666;
    margin-bottom: 1.5rem;
}
.winner-box {
    background: linear-gradient(135deg, #2E7D32 0%, #388E3C 100%);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 1.3rem;
    font-weight: bold;
    margin: 0.5rem 0;
}
.metric-card {
    background: #f0f4ff;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1A237E;
    margin: 0.5rem 0;
}
.section-header {
    color: #1A237E;
    font-weight: bold;
    font-size: 1.1rem;
    border-bottom: 2px solid #FF6F00;
    padding-bottom: 4px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────
MODEL_PATH  = "model/"
DATA_PATH   = "data/processed/"
PARTY_COLORS = {
    "BJP":     "#FF6F00",
    "Congress":"#1565C0",
    "JDS":     "#2E7D32",
    "general": "#607D8B"
}

# ── HELPER FUNCTIONS ───────────────────────────────────────────
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

@st.cache_data
def load_data():
    path = os.path.join(DATA_PATH, "cleaned_dataset.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_results():
    path = os.path.join(DATA_PATH, "prediction_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            model.eval()
            return tokenizer, model
        except Exception as e:
            st.warning(f"Model not found locally. Using VADER for predictions. ({e})")
            return None, None
    return None, None

def predict_sentiment(text, tokenizer, model):
    if tokenizer is None or model is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        score    = analyzer.polarity_scores(str(text))["compound"]
        if score >= 0.05:    return "Positive", [0.1, 0.2, 0.7]
        elif score <= -0.05: return "Negative", [0.7, 0.2, 0.1]
        else:                return "Neutral",  [0.1, 0.8, 0.1]

    inputs = tokenizer(
        str(text), return_tensors="pt",
        max_length=128, truncation=True, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs  = torch.softmax(outputs.logits, dim=1).numpy()[0]
    labels = ["Negative", "Neutral", "Positive"]
    return labels[np.argmax(probs)], probs.tolist()

# ── COLLECT FRESH DATA ─────────────────────────────────────────
def collect_fresh_news():
    articles = []
    queries  = [
        "Karnataka election BJP",
        "Karnataka election Congress",
        "Siddaramaiah Karnataka",
        "JDS Karnataka election",
        "Kumaraswamy Karnataka",
    ]
    for query in queries:
        try:
            url      = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
            soup     = BeautifulSoup(response.text, "xml")
            items    = soup.find_all("item")
            for item in items[:10]:
                title       = item.find("title")
                description = item.find("description")
                text        = ""
                if title:       text += title.get_text(strip=True)
                if description: text += " " + description.get_text(strip=True)
                if text.strip():
                    articles.append({
                        "text":         text.strip(),
                        "cleaned_text": clean_text(text.strip()),
                        "party":        assign_party(text),
                        "platform":     "google_news",
                        "query_used":   query,
                        "collected_at": datetime.now().isoformat()
                    })
            time.sleep(0.5)
        except Exception:
            pass
    return pd.DataFrame(articles)

# ── LOAD DATA ──────────────────────────────────────────────────
df      = load_data()
results = load_results()

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗳️ Karnataka Election")
    st.markdown("**Sentiment Analysis System**")
    st.markdown("---")

    st.markdown("### 🎛️ Filters")
    parties   = ["All"] + [p for p in ["BJP", "Congress", "JDS", "general"] if not df.empty and p in df["party"].unique()]
    platforms = ["All"] + (list(df["platform"].unique()) if not df.empty else [])

    selected_party    = st.selectbox("Party", parties)
    selected_platform = st.selectbox("Platform", platforms)

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("**Model:** XLM-RoBERTa")
    st.markdown("**Accuracy:** 77%")
    st.markdown("**Languages:** English + Kannada")
    st.markdown("**Platforms:** 6")
    st.markdown("---")
    st.markdown("### 👩‍💻 Project Info")
    st.markdown("**Student:** Pooja K V")
    st.markdown("**USN:** 24MTRCS007")
    st.markdown("**Domain:** MTech CSE")
    st.markdown("**Year:** 2024-25")

# ── TITLE ──────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🗳️ Karnataka Election Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Cross-Platform Deep Learning Based Election Prediction System | XLM-RoBERTa | Pooja K V</div>", unsafe_allow_html=True)
st.markdown("---")

# ── FILTER DATA ────────────────────────────────────────────────
if not df.empty:
    filtered_df = df.copy()
    if selected_party != "All":
        filtered_df = filtered_df[filtered_df["party"] == selected_party]
    if selected_platform != "All":
        filtered_df = filtered_df[filtered_df["platform"] == selected_platform]
else:
    filtered_df = pd.DataFrame()

# ── TAB LAYOUT ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏆 Prediction",
    "📈 Sentiment Analysis",
    "🌐 Platform Analysis",
    "🤖 Live Predictor"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📊 Dataset Overview")

    if not df.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Records",    len(df))
        col2.metric("BJP Records",      len(df[df.party=="BJP"]) if "party" in df.columns else 0)
        col3.metric("Congress Records", len(df[df.party=="Congress"]) if "party" in df.columns else 0)
        col4.metric("JDS Records",      len(df[df.party=="JDS"]) if "party" in df.columns else 0)
        col5.metric("Platforms",        df["platform"].nunique() if "platform" in df.columns else 0)
    else:
        st.warning("No data found. Please ensure data/processed/cleaned_dataset.csv exists.")

    st.markdown("---")
    st.markdown("### 🔄 System Pipeline")

    stages = [
        ("📥 Data Collection",   "6 platforms\n1,218 records"),
        ("🧹 Preprocessing",     "Clean + Label\n1,063 records"),
        ("🏷️ Sentiment Labels",  "VADER + Sentiment140\n3,063 records"),
        ("🧠 XLM-RoBERTa",       "Fine-tuning\n77% accuracy"),
        ("🗳️ Prediction",        "Party scoring\nElection winner"),
    ]

    cols = st.columns(5)
    for i, (title, desc) in enumerate(stages):
        with cols[i]:
            st.markdown(f"""
            <div style='background:#1A237E;color:white;padding:12px;border-radius:8px;text-align:center;min-height:90px'>
                <b>{title}</b><br><small style='color:#CADCFC'>{desc.replace(chr(10),'<br>')}</small>
            </div>
            """, unsafe_allow_html=True)
            if i < 4:
                pass

    st.markdown("---")

    if not df.empty and "party" in df.columns:
        st.markdown("### 📋 Sample Data")
        party_filter = df[df["party"].isin(["BJP","Congress","JDS"])][["text","party","platform"]].head(10)
        st.dataframe(party_filter, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏆 Election Prediction Results")

    if not results.empty:
        col1, col2 = st.columns([1, 2])

        with col1:
            winner = results["avg_score"].idxmax() if "avg_score" in results.columns else "JDS"
            st.markdown(f"""
            <div class='winner-box'>
                🏆 PREDICTED WINNER<br><br>
                <span style='font-size:2.5rem'>{winner}</span><br><br>
                <small>Based on cross-platform sentiment analysis</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Sentiment Scores")
            for party in ["BJP","Congress","JDS"]:
                if party in results.index and "avg_score" in results.columns:
                    score = results.loc[party, "avg_score"]
                    color = PARTY_COLORS.get(party, "#607D8B")
                    icon  = "🥇" if party == winner else ("🥈" if results["avg_score"].rank(ascending=False)[party] == 2 else "🥉")
                    st.markdown(f"""
                    <div style='background:#f8f9fa;padding:8px 12px;border-radius:8px;
                                border-left:4px solid {color};margin:6px 0'>
                        {icon} <b style='color:{color}'>{party}</b>
                        <span style='float:right;color:#333'>{score:+.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            if "positive_pct" in results.columns:
                fig = go.Figure()
                for party in ["BJP","Congress","JDS"]:
                    if party in results.index:
                        fig.add_trace(go.Bar(
                            name=party,
                            x=[party],
                            y=[results.loc[party,"positive_pct"]],
                            marker_color=PARTY_COLORS.get(party,"#888"),
                            text=[f"{results.loc[party,'positive_pct']:.1f}%"],
                            textposition="outside",
                            textfont=dict(size=14, color="#333")
                        ))
                fig.update_layout(
                    title="Positive Sentiment % by Party",
                    yaxis_title="Positive Sentiment %",
                    yaxis=dict(range=[0,50]),
                    showlegend=False,
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Prediction results not found. Please run the model prediction script first.")

        st.markdown("#### 📌 Manual Results (from last run)")
        manual = {
            "Party":   ["BJP",    "Congress", "JDS"],
            "Score":   [-0.067,   -0.006,     +0.043],
            "Positive":[86,       94,         29],
            "Negative":[112,      96,         24],
            "Neutral": [191,      164,        62],
            "Rank":    ["3rd 🥉", "2nd 🥈",  "1st 🥇"],
        }
        st.dataframe(pd.DataFrame(manual), use_container_width=True)

        st.markdown("""
        <div class='winner-box'>
            🏆 Predicted Winner: JDS (Janata Dal Secular)<br>
            Sentiment Score: +0.043
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — SENTIMENT ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Sentiment Breakdown")

    if not results.empty and all(col in results.columns for col in ["positive","negative","neutral"]):
        col1, col2 = st.columns(2)

        with col1:
            parties   = [p for p in ["BJP","Congress","JDS"] if p in results.index]
            positives = [results.loc[p,"positive"] for p in parties]
            negatives = [results.loc[p,"negative"] for p in parties]
            neutrals  = [results.loc[p,"neutral"]  for p in parties]

            fig2 = go.Figure(data=[
                go.Bar(name="Positive", x=parties, y=positives, marker_color="#2E7D32"),
                go.Bar(name="Negative", x=parties, y=negatives, marker_color="#C62828"),
                go.Bar(name="Neutral",  x=parties, y=neutrals,  marker_color="#607D8B"),
            ])
            fig2.update_layout(
                barmode="group", title="Sentiment Count by Party",
                height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            if not df.empty and "party" in df.columns:
                party_data = df[df["party"].isin(["BJP","Congress","JDS"])]["party"].value_counts()
                fig3 = px.pie(
                    values=party_data.values, names=party_data.index,
                    title="Data Distribution by Party",
                    color=party_data.index,
                    color_discrete_map=PARTY_COLORS, height=380
                )
                st.plotly_chart(fig3, use_container_width=True)

    else:
        st.info("Run prediction first to see sentiment breakdown.")
        # Show static chart from hardcoded data
        parties   = ["BJP", "Congress", "JDS"]
        positives = [86, 94, 29]
        negatives = [112, 96, 24]
        neutrals  = [191, 164, 62]

        fig2 = go.Figure(data=[
            go.Bar(name="Positive", x=parties, y=positives, marker_color="#2E7D32"),
            go.Bar(name="Negative", x=parties, y=negatives, marker_color="#C62828"),
            go.Bar(name="Neutral",  x=parties, y=neutrals,  marker_color="#607D8B"),
        ])
        fig2.update_layout(
            barmode="group", title="Sentiment Count by Party (Last Run Results)",
            height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    perf_data = [
        {"Round": "Round 1", "Model": "mBERT + TextBlob",          "Accuracy": 64.79},
        {"Round": "Round 2", "Model": "XLM-RoBERTa + VADER",       "Accuracy": 70.42},
        {"Round": "Round 3", "Model": "XLM-RoBERTa + Combined",    "Accuracy": 77.00},
    ]

    fig_perf = go.Figure(go.Bar(
        x=[d["Round"] for d in perf_data],
        y=[d["Accuracy"] for d in perf_data],
        text=[f"{d['Accuracy']}%" for d in perf_data],
        textposition="outside",
        marker_color=["#607D8B","#1565C0","#2E7D32"],
    ))
    fig_perf.update_layout(
        title="Model Accuracy Progression",
        yaxis=dict(range=[55, 85], title="Accuracy %"),
        height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_perf, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PLATFORM ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🌐 Platform-wise Analysis")

    if not df.empty and "platform" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            platform_counts = df["platform"].value_counts()
            fig4 = px.bar(
                x=platform_counts.index, y=platform_counts.values,
                title="Records by Platform",
                color=platform_counts.values,
                color_continuous_scale="Blues",
                height=380, labels={"x": "Platform", "y": "Records"}
            )
            fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)

        with col2:
            if "party" in df.columns:
                party_platform = pd.crosstab(df["platform"], df["party"])
                fig5 = px.imshow(
                    party_platform, title="Party Mentions by Platform",
                    color_continuous_scale="Blues", height=380
                )
                st.plotly_chart(fig5, use_container_width=True)

        st.markdown("### 📋 Platform Summary Table")
        summary = df.groupby("platform").agg(
            Records=("text","count"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)

    else:
        st.info("No data loaded. Showing static platform summary.")
        static = pd.DataFrame({
            "Platform":     ["YouTube","Google News","The Hindu","NDTV","Times of India","Wikipedia"],
            "Records":      [314, 680, 91, 6, 8, 119],
            "Type":         ["Comments","Articles","Articles","Articles","Articles","Encyclopedia"],
            "Language":     ["EN+KN","EN","EN","EN","EN","EN"],
        })
        st.dataframe(static, use_container_width=True)

        fig_static = px.bar(
            static, x="Platform", y="Records",
            title="Records Collected by Platform",
            color="Records", color_continuous_scale="Blues", height=380
        )
        fig_static.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_static, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🤖 Live Sentiment Predictor")
    st.markdown("Type any text about Karnataka politics and get instant AI-powered sentiment prediction!")

    col1, col2 = st.columns([2, 1])

    with col1:
        user_input = st.text_area(
            "Enter text here:",
            placeholder="e.g. BJP is doing great work in Karnataka infrastructure development...",
            height=120
        )

        examples = [
            "BJP is doing excellent work in Karnataka",
            "Congress has failed to deliver on promises",
            "JDS is a strong regional party in Karnataka",
            "ಕರ್ನಾಟಕದಲ್ಲಿ ಚುನಾವಣೆ ಬಹಳ ಮುಖ್ಯ",
        ]
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(2)
        for i, ex in enumerate(examples):
            if ex_cols[i % 2].button(ex[:40] + "...", key=f"ex_{i}"):
                user_input = ex

    with col2:
        st.markdown("#### 🎯 How it works")
        st.markdown("""
        1. Enter political text
        2. XLM-RoBERTa model analyzes it
        3. Returns sentiment probabilities
        4. Shows positive/negative/neutral
        """)

    if st.button("🔍 Predict Sentiment", type="primary", use_container_width=True):
        if user_input and user_input.strip():
            with st.spinner("Analyzing sentiment with XLM-RoBERTa..."):
                tokenizer, model = load_model()
                sentiment, probs = predict_sentiment(user_input, tokenizer, model)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 Negative", f"{probs[0]*100:.1f}%")
            col2.metric("⚪ Neutral",  f"{probs[1]*100:.1f}%")
            col3.metric("🟢 Positive", f"{probs[2]*100:.1f}%")

            if sentiment == "Positive":
                st.success(f"✅ Predicted Sentiment: **POSITIVE**")
            elif sentiment == "Negative":
                st.error(f"❌ Predicted Sentiment: **NEGATIVE**")
            else:
                st.info(f"➖ Predicted Sentiment: **NEUTRAL**")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probs[2] * 100,
                title={"text": "Positivity Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#2E7D32"},
                    "steps": [
                        {"range": [0, 33],  "color": "#FFCDD2"},
                        {"range": [33, 66], "color": "#FFF9C4"},
                        {"range": [66, 100],"color": "#C8E6C9"},
                    ],
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Party detection
            detected_party = assign_party(user_input)
            if detected_party != "general":
                color = PARTY_COLORS.get(detected_party, "#607D8B")
                st.markdown(f"""
                <div style='background:#f0f4ff;padding:10px;border-radius:8px;
                            border-left:4px solid {color};margin-top:10px'>
                    🏷️ <b>Party Detected:</b>
                    <span style='color:{color};font-weight:bold'> {detected_party}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze!")

    st.markdown("---")
    st.markdown("### 📰 Collect Fresh News")
    if st.button("🔄 Fetch Latest Karnataka Politics News"):
        with st.spinner("Collecting fresh news from Google News RSS..."):
            fresh_df = collect_fresh_news()
        if not fresh_df.empty:
            st.success(f"✅ Collected {len(fresh_df)} fresh articles!")
            st.dataframe(fresh_df[["text","party","platform"]].head(10), use_container_width=True)
        else:
            st.warning("Could not fetch news. Check internet connection.")

# ── FOOTER ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.85rem'>
    Built by <b>Pooja K V</b> | MTech Project 2024-25 |
    Cross-Platform Sentiment Analysis using Deep Learning |
    XLM-RoBERTa | Streamlit
</div>
""", unsafe_allow_html=True)
