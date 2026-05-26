# AI-Based Career Path Prediction and Guidance System

Overview

The AI-Based Career Path Prediction and Guidance System is a Machine Learning-based web application developed to help students identify suitable career paths based on their skills, interests, academic performance, certifications, and personality traits. The system predicts appropriate job roles and provides career guidance using Machine Learning algorithms.

Features

Career prediction using Machine Learning
Personalized career recommendations
Course recommendation system
Interactive Streamlit web interface
Data preprocessing and analysis
Model evaluation and comparison
User-friendly dashboard
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib
Seaborn
XGBoost
Pickle
Machine Learning Models Used
Random Forest Classifier
Decision Tree Classifier
XGBoost Classifier
Project Structure
AI-Career-Guidance-System/
│
├── app.py
├── train_model.py
├── utils.py
├── PS2_Dataset.csv
├── career_model.pkl
├── requirements.txt
└── README.md

Dataset

The project uses a dataset containing:

Skills
Certifications
Workshops
Academic information
Personality traits
Communication skills
Interests

Target Column:

Suggested Job Role
Installation
Clone Repository
git clone <repository-link>
cd AI-Career-Guidance-System
Install Required Libraries
pip install -r requirements.txt
Run the Project
Train the Model
python train_model.py
Run Streamlit Application
streamlit run app.py
System Workflow
User Input
    ↓
Data Preprocessing
    ↓
Machine Learning Prediction
    ↓
Career Recommendation
    ↓
Course Suggestions
Future Scope
Resume Analyzer
AI Chatbot Integration
Cloud Deployment
Student Analytics Dashboard
Real-time Recommendation System
Conclusion

This project demonstrates the practical implementation of Artificial Intelligence and Machine Learning in career guidance systems. It helps students make informed career decisions through intelligent and personalized recommendations.

Author:
Provish Kumar

License:
This project is developed for educational and internship purposes.
