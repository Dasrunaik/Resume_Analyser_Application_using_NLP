# Resume_Screnning_Application_using_NLP


This project demonstrates an intelligent resume screening system using Natural Language Processing (NLP) and Machine Learning (ML) to automatically classify resumes into specific job categories. It aims to assist recruiters and HR teams in efficiently shortlisting candidates.

#🚀 Features

Classifies resumes into 25 job categories (e.g., Data Science, Java Developer, HR, Testing).

Built with Streamlit for an interactive web UI.

Utilizes TF-IDF vectorization and a trained K-Nearest Neighbors (KNN) classifier.

Text preprocessing with NLTK: tokenization, stopword removal, regex cleaning.

Accepts .pdf and .txt resume files for prediction.

Real-time resume screening with category output.

#🛠️ Tech Stack

Python

NLTK

scikit-learn

pandas, NumPy

Streamlit

Pickle (for model serialization)

#🧠 Machine Learning Workflow

Data Loading: Used a labeled dataset of ~900 resumes.

Text Cleaning: Removed URLs, punctuation, special characters.

Feature Extraction: Applied TF-IDF vectorizer.

Model Training: KNN classifier trained and evaluated.

Deployment: Streamlit used to create a web interface for resume upload and prediction.


#📂 Folder Structure

📁 Resume_Screening_NLP

├── resume_screening_using_nlp.py      # Main Streamlit app

├── clf.pkl                            # Trained KNN classifier

├── tfidf.pkl                          # Saved TF-IDF vectorizer

├── resume_dataset.csv                 # Resume data (if available)

├── README.md                          # Project documentation


#🔍 Sample Categories

Data Science

Java Developer

HR

Web Designing

DevOps Engineer

Python Developer
...and many more (25 total)

#📈 Results

Model Accuracy: ~98% on test data

Fast Predictions: <1 sec per resume

Real-Time Classification via UI

#📌 Future Enhancements

Add OCR to support image-based resumes

Expand to multilingual resume parsing

Deploy to Hugging Face Spaces or AWS Lambda

