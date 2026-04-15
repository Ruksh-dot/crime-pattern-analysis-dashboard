🚓 Crime Pattern Analysis Dashboard

📊 Overview

This project is an end-to-end Crime Pattern Analysis Dashboard built using Machine Learning and Streamlit. It analyzes crime data across geographical, temporal, and feature dimensions to uncover patterns and support data-driven decision-making.

The system focuses on identifying crime hotspots, peak crime periods, and key influencing factors, and translates them into actionable insights.



🚀 Key Features

🌍 Geo Analysis
	•	KMeans clustering to identify crime hotspots
	•	District and beat-level insights
	•	Location-based risk understanding

⏱ Temporal Analysis
	•	Crime trends by hour and month
	•	Peak crime detection
	•	Smart patrol planning system
	•	Risk-based recommendations

📉 PCA Insights
	•	Dimensionality reduction using PCA
	•	Feature contribution analysis
	•	Identification of dominant influencing factors


🧠 Key Learnings
	•	Applied unsupervised learning techniques (KMeans clustering)
	•	Understood and interpreted Silhouette Score for clustering performance
	•	Learned practical usage of PCA for dimensionality reduction
	•	Built a complete data pipeline from preprocessing → modeling → visualization
	•	Designed a user-friendly interactive dashboard using Streamlit
	•	Structured a real-world project with proper modular architecture



💼 Business Impact
	•	Identifies high-risk crime zones
	•	Helps optimize patrol deployment
	•	Improves response time using temporal insights
	•	Enables data-driven policing strategies
	•	Demonstrates real-world application of ML in public safety


🛠 Tech Stack
	•	Python
	•	Streamlit
	•	Pandas, NumPy
	•	Scikit-learn (KMeans, PCA)
	•	Matplotlib

📂 Project Structure

    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── models/
    │   ├── geo_scaler.pkl
    │   ├── temporal_scaler.pkl
    │   ├── pca_model.pkl
    │   └── pca_scaler.pkl


⚠️ Dataset Note

   Due to large file sizes, datasets are not stored directly in the repository.
   They are loaded dynamically from external storage (Google Drive) during runtime.

   Link: https://drive.google.com/drive/folders/18iffyqG37gJdGQzVi0tmk8iHGC4kcVE-?usp=drive_link

