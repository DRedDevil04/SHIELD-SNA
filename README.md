# 🛡 S.H.I.E.L.D. – Social Hoax Identification & Event Linkage Dashboard

## 📌 Overview
**S.H.I.E.L.D.** (*Social Hoax Identification & Event Linkage Dashboard*) is a standalone system to detect, analyze, and visualize **hoax call campaigns** on Reddit using **NLP**, **network analysis**, **sentiment evaluation**, and **temporal correlation** techniques.

---

## 🚀 Quickstart Guide

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/DRedDevil04/SHIELD-SNA.git
cd SHIELD-SNA
```
### ✅ 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### ✅ 3. Configure Reddit API
Create a Reddit App at https://www.reddit.com/prefs/apps and add credentials to a .env file:

```ini
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```
### 📊 Workflow
#### 📥 Step 1: Data Collection
Run the notebook to extract Reddit posts:

```bash
reddit_collect.ipynb
```
#### 🏗 Step 2: Data Construction
Structure and preprocess the collected data:

```bash
construct_data.ipynb
```
#### 🚀 Step 3: Launch the Dashboard
Run the full analytical dashboard:

```bash
python3 app2.py
```
### 🧠 Features
#### 🕵️‍♂️ Hoax Detection using NLP and ML

#### 🌐 Author Network Graphs with Plotly + NetworkX

#### ⏰ Temporal Correlation with real-world hoax events

#### 🧠 Sentiment & Threat Analysis

#### 📈 Fully Integrated Visualization Dashboard

### ⚙️ Tech Stack
## 🧰 Tech Stack Summary

| **Category**         | **Tools Used**                           |
|----------------------|-------------------------------------------|
| **Language**         | Python 3.9+                               |
| **Data Collection**  | Pushshift API, PRAW                      |
| **NLP & ML**         | NLTK, SpaCy, Scikit-learn                |
| **Sentiment**        | TextBlob, VADER                          |
| **Network Analysis** | NetworkX, Plotly                         |
| **Dashboard**        | Dash, Matplotlib, Seaborn                |

### 📦 Output
#### ✅ Hoax Classification Report

#### ✅ User Interaction Graph

#### ✅ Sentiment Trend Analysis

#### ✅ Temporal Event Correlation

#### ✅ Threat Level Insights

### 🤝 Contributors
#### _**Devam Desai (IIT2022035)**_
#### _**Nitu Sherawat (IIT2022073)**_
#### **_Samar Borkar (IIT2022083)_**
