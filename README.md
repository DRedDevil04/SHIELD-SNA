# **S.H.I.E.L.D. – Social Hoax Identification & Event Linkage Dashboard**

## 📌 Overview
**S.H.I.E.L.D.** (*Social Hoax Identification & Event Linkage Dashboard*) is a **standalone system** designed to automate the detection, analysis, and visualization of social media data related to **hoax call campaigns**. The system uses a combination of **network analysis**, **content analysis**, and **temporal correlation** to map hoax-related communication, assess threats, and visualize the spread of hoaxes on social media platforms like **Reddit**.

## 🎯 Features
✅ **Multi-platform Data Collection** – Extracts data from Reddit using **Pushshift.io API** and the **PRAW library**.  
✅ **Content Analysis Module** – Identifies hoax-related posts using **NLP-based techniques** and **machine learning** algorithms.  
✅ **User Network Mapping** – Detects coordinated efforts in spreading hoaxes using **network analysis** techniques.  
✅ **Temporal Correlation Analysis** – Links hoax-related posts to **real-world hoax call events** over time.  
✅ **Sentiment & Threat Assessment** – Analyzes the sentiment of posts and assesses their potential real-world impact.  
✅ **Social Network Visualization** – Graphical representation of **how hoax content spreads** across users.  
✅ **Validation Report** – Provides case studies and insights into real-world hoax incidents.  

## 🛠️ Tech Stack & Dependencies
### **Programming Language:**
- **Python 3.9+** – Core development language

### **Data Collection Tools:**
- **Pushshift.io API** – Extracts Reddit data
- **PRAW** – Reddit API Wrapper for live data collection

### **Text Processing & NLP:**
- **NLTK / SpaCy** – Tokenization, lemmatization, and Named Entity Recognition (NER)
- **TextBlob / VADER** – Sentiment analysis
- **Scikit-learn** – Machine learning-based classification and analysis

### **Network & Visualization Tools:**
- **NetworkX** – Graph-based analysis of user interactions
- **Plotly** – Interactive network visualizations
- **Matplotlib / Seaborn** – Data visualization
- **Gephi** – Standalone network visualization tool

### **Datasets Used:**
- **Pushshift.io Reddit Dumps** – Archived Reddit data for hoax analysis

## 🔧 Installation & Setup
### **1️⃣ Clone the Repository:**
```bash
git clone https://github.com/yourusername/SHIELD.git
cd SHIELD
```
### **2️⃣ Install Dependencies:**
```bash
pip install -r requirements.txt
```

### ** 3️⃣ Set Up API Keys**
```bash
Reddit API: Create an app at Reddit Developer Portal

Create a .env file and add:
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

### ** 4️⃣ Run the System:**
```bash
python3 app.py
```

📌 Workflow
1️⃣ Data Collection – Extract hoax-related posts from Reddit using Pushshift.io and PRAW.
2️⃣ Content Analysis – Identifies hoax indicators and misinformation using NLP techniques and machine learning models.
3️⃣ User Network Mapping – Visualizes coordinated efforts of users spreading hoaxes through network analysis.
4️⃣ Temporal Analysis – Links social media posts to real-world hoax events using timestamps and event data.
5️⃣ Sentiment & Threat Analysis – Determines the impact and seriousness of hoax messages.
6️⃣ Visualization & Reporting – Generates network graphs, temporal correlation plots, and threat assessment reports.

📊 Expected Output
📌 Hoax Probability Score – Classifies posts as Hoax or Not based on machine learning analysis.
📌 User Coordination Graph – Visualizes connections between users spreading hoaxes using network graphs.
📌 Sentiment Analysis Chart – Displays trends of positive, neutral, and negative sentiments in posts.
📌 Temporal Correlation Plot – Links hoax posts with real-world event timelines.
📌 Threat Assessment Report – Evaluates if a hoax could lead to real-world harm or disruptions.

🏆 Use Cases
Law enforcement agencies – Detect and mitigate hoax call campaigns and social media misinformation.

Social media platforms – Flag and take down hoax-related content.

Crisis management teams – Quickly identify and respond to false alarm incidents.

Journalists & Fact-checkers – Analyze trends in social media misinformation and hoaxes.

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributors
DEVAM DESAI
NITU SHERAWAT
SAMAR BORKAR

📬 Contact
For questions, feedback, or collaborations, reach out at example.com or create an issue on GitHub.

