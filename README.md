# VidMetrics: Video Engagement Analysis & Prediction Platform

**VidMetrics** is an end-to-end data analytics and machine learning solution designed to quantify, evaluate, and predict video engagement scores based on video attributes and metadata. The platform pairs a robust `scikit-learn` machine learning pipeline with an interactive multi-page `Streamlit` web application (`app_2.py`).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset Architecture](#dataset-architecture)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [Streamlit Application Structure](#streamlit-application-structure)
- [Installation & Quick Start](#installation--quick-start)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## 🔍 Overview

Predicting video engagement is critical for content creators, digital marketers, and media strategists. VidMetrics leverages ensemble machine learning models to identify non-linear relationships across metadata—such as duration, likes, comments, title length, publish hour, and category—to estimate engagement scores in real time.

---

## ✨ Key Features

* **Interactive Data Exploration (EDA):** Dynamic dataset filtering, summary statistics, feature correlation heatmaps, and category-level distribution analyses.
* **Real-Time Prediction Engine:** User-friendly control panel allowing custom input parameters to calculate real-time predicted engagement scores accompanied by automated tier classifications:
  * 🟢 **High Engagement:** $> 8.0\%$
  * 🟡 **Medium Engagement:** $4.0\% - 8.0\%$
  * 🔴 **Low Engagement:** $< 4.0\%$
* **Model Performance Diagnostics:** In-depth evaluation metrics ($R^2$, $RMSE$, $MAE$), residual diagnostics, and tree-based feature importance rankings.

---

## 📁 Repository Structure

```text
.
├── app_2.py                         # Interactive Streamlit Web Application
├── best_video_engagement_model.pkl  # Serialized Machine Learning Pipeline
├── requirements.txt                 # Python Dependencies
└── README.md                        # Project Documentation
```

---

## 📊 Dataset Architecture

The underlying dataset contains **2,000 video records** across 8 independent predictor variables and 1 continuous response metric (`engagement_score`).

### Feature Dictionary

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `duration_sec` | Numerical | Video length in seconds (30s – 1,798s) |
| `views` | Numerical | Total impression/view count |
| `likes` | Numerical | Total like count |
| `comments` | Numerical | Total comment count |
| `title_len` | Numerical | Character length of the video title |
| `tags_count` | Numerical | Total number of tags utilized |
| `publish_hour` | Numerical | Hour of publication in 24-hr format (0 – 23) |
| `category` | Categorical | Video category (`Education`, `Tech`, `Gaming`, `Music`, `Entertainment`, `Vlogs`) |
| **`engagement_score`** | **Target** | Bounded target engagement metric (0.50% – 20.00%) |

---

## 📈 Model Performance & Evaluation

Three regression algorithms were trained and tested using an **80/20 train/test split** ($N_{train}=1,600$, $N_{test}=400$):

| Model Algorithm | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | 0.5419 | 0.4964 | 3.4760 | 2.6374 |
| **Random Forest Regressor** | 0.9852 | **0.9038** | 1.5189 | 1.1598 |
| **Gradient Boosting Regressor** | 0.9525 | **0.9016** | 1.5364 | 1.1794 |

> **Key Insight:** Ensemble tree algorithms (Gradient Boosting and Random Forest) outperform linear models ($R^2 > 0.90$), effectively capturing non-linear interactions between viewer metrics and video duration.

---

## 🚀 Streamlit Application Structure

The web application (`app_2.py`) is organized into three interactive pages:

1. **Page 1: Dataset & EDA Dashboard**
   * Raw dataset preview and descriptive statistics.
   * Seaborn correlation heatmaps and boxplots by video category.
2. **Page 2: Real-Time Engagement Predictor**
   * Sidebar controls for adjusting video metadata attributes.
   * Real-time model inference showing predicted percentage score and tier badge.
3. **Page 3: Model Performance & Diagnostics**
   * Model benchmark tables and metrics comparison ($R^2$, $RMSE$, $MAE$).
   * Dynamic feature importance visualization.

---

## 🛠️ Installation & Quick Start

### Prerequisites

Ensure you have **Python 3.8+** installed on your machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/vidmetrics.git
cd vidmetrics
```

### Step 2: Set Up Virtual Environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app_2.py
```

The app will launch automatically in your browser at `http://localhost:8501`.

---

## 💻 Technologies Used

* **Language:** Python 3.8+
* **Machine Learning:** `scikit-learn`
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Web Framework:** `Streamlit`
* **Model Serialization:** `joblib` / `pickle`

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.# Video-Engagement-Analysis
