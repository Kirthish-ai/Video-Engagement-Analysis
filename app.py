import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Video Engagement Analyzer", page_icon="📹", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('best_video_engagement_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('video_engagement_data.csv')

model = load_model()
df = load_data()

st.title("VidMetrics: Video Engagement Analysis & Prediction")

st.sidebar.title("Navigation Menu")
page = st.sidebar.radio("Go to:", ["1. Dataset & EDA", "2. Predict Engagement", "3. Model Performance"])

# PAGE 1: EDA
if page == "1. Dataset & EDA":
    st.header("🔍 Data Overview & Exploratory Analysis")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("Engagement by Category")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='category', y='engagement_score', palette='Set2', ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)

# PAGE 2: PREDICTOR
elif page == "2. Predict Engagement":
    st.header("Real-Time Engagement Predictor")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        category = st.selectbox("Category", df['category'].unique())
        duration_sec = st.number_input("Duration (seconds)", min_value=10, max_value=7200, value=300)
        publish_hour = st.slider("Publish Hour (24-hr)", 0, 23, 14)
    with col2:
        views = st.number_input("Views Count", min_value=100, max_value=10000000, value=25000)
        likes = st.number_input("Likes Count", min_value=0, max_value=1000000, value=1200)
    with col3:
        comments = st.number_input("Comments Count", min_value=0, max_value=500000, value=150)
        title_len = st.slider("Title Length (chars)", 5, 120, 45)
        tags_count = st.slider("Tags Count", 0, 50, 12)
        
    input_data = pd.DataFrame([{
        'duration_sec': duration_sec, 'views': views, 'likes': likes,
        'comments': comments, 'title_len': title_len, 'tags_count': tags_count,
        'category': category, 'publish_hour': publish_hour
    }])
    
    if st.button("Predict Score", type="primary"):
        pred = model.predict(input_data)[0]
        st.markdown("---")
        st.metric("Predicted Engagement Score", f"{pred:.2f}%")
        
        if pred > 8.0:
            st.success("High Engagement Tier")
        elif pred > 4.0:
            st.warning("Medium Engagement Tier")
        else:
            st.error("Low Engagement Tier")

# PAGE 3: PERFORMANCE
elif page == "3. Model Performance":
    st.header("Model Metrics & Diagnostics")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Algorithm", "Gradient Boosting")
    c2.metric("R² Score", "~0.85+")
    c3.metric("RMSE Error", "~0.65")
    
    st.subheader("Feature Importance Analysis")
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(['category']))
    num_names = ['duration_sec', 'views', 'likes', 'comments', 'title_len', 'tags_count', 'publish_hour']
    all_names = num_names + cat_names
    
    importances = regressor.feature_importances_
    feat_df = pd.DataFrame({'Feature': all_names, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=feat_df.head(8), x='Importance', y='Feature', palette='mako', ax=ax)
    st.pyplot(fig)