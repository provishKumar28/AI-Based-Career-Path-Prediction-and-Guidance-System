import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Reconfigure stdout to use UTF-8 to prevent console encoding crashes on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure src folder is searchable for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_preprocessing import clean_column_names, execute_preprocessing
from model_training import CareerMLP, train_and_evaluate
from eda_script import generate_eda_reports

# Directory configuration
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

# Course recommendations database mappingpredicted careers
COURSE_RECOMMENDATIONS = {
    'Applications Developer': [
        {"title": "Java Programming & Software Engineering Fundamentals", "platform": "Coursera (Duke University)"},
        {"title": "Python for Application Development", "platform": "edX (IBM)"},
        {"title": "Object-Oriented Programming & Systems Designing Masterclass", "platform": "Udemy"}
    ],
    'CRM Technical Developer': [
        {"title": "Salesforce Certified Platform Developer I Preparation", "platform": "Udemy"},
        {"title": "Microsoft Dynamics 365 Core Developer Course", "platform": "Microsoft Learn"},
        {"title": "Introduction to Customer Relationship Management (CRM)", "platform": "edX"}
    ],
    'Database Developer': [
        {"title": "Database Design & SQL Specialization", "platform": "Coursera (University of Michigan)"},
        {"title": "Oracle Database SQL Certified Associate Course", "platform": "Udemy"},
        {"title": "SQL & Relational Databases 101", "platform": "cognitiveclass.ai"}
    ],
    'Mobile Applications Developer': [
        {"title": "iOS & Swift - Complete App Development Bootcamp", "platform": "Udemy (Angela Yu)"},
        {"title": "Android App Development Specialization", "platform": "Coursera (Vanderbilt University)"},
        {"title": "React Native - The Practical Guide", "platform": "Udemy (Academind)"}
    ],
    'Network Security Engineer': [
        {"title": "Google Cybersecurity Professional Certificate", "platform": "Coursera"},
        {"title": "CompTIA Security+ Certification Exam Preparation", "platform": "Udemy"},
        {"title": "Introduction to IT & Cybersecurity Specialization", "platform": "Cybrary"}
    ],
    'Software Developer': [
        {"title": "CS50's Introduction to Computer Science", "platform": "edX (Harvard University)"},
        {"title": "Software Design and Architecture Specialization", "platform": "Coursera (University of Alberta)"},
        {"title": "The Complete Software Developer's Career Guide", "platform": "Udemy"}
    ],
    'Software Engineer': [
        {"title": "Data Structures and Algorithms Specialization", "platform": "Coursera (UC San Diego)"},
        {"title": "Software Architecture and Design Patterns Masterclass", "platform": "Udemy"},
        {"title": "System Design for High Performance & Scalability", "platform": "Educative.io"}
    ],
    'Software Quality Assurance (QA) / Testing': [
        {"title": "Software Testing & QA Professional Certificate", "platform": "Coursera (University of Minnesota)"},
        {"title": "Selenium WebDriver with Java - Basics to Advanced", "platform": "Udemy (Rahul Shetty)"},
        {"title": "Automated Software Testing & DevOps", "platform": "edX"}
    ],
    'Systems Security Administrator': [
        {"title": "System Administration & IT Infrastructure Services", "platform": "Coursera (Google)"},
        {"title": "CISSP Certified Information Systems Security Exam Prep", "platform": "Udemy"},
        {"title": "Certified Information Security Manager (CISM)", "platform": "edX"}
    ],
    'Technical Support': [
        {"title": "Google IT Support Professional Certificate", "platform": "Coursera"},
        {"title": "CompTIA A+ Core Certification Prep", "platform": "Udemy"},
        {"title": "Technical Support Fundamentals", "platform": "Coursera"}
    ],
    'UX Designer': [
        {"title": "Google UX Design Professional Certificate", "platform": "Coursera"},
        {"title": "User Experience Design Essentials - Adobe XD", "platform": "Udemy"},
        {"title": "Interaction Design Specialist Course", "platform": "Interaction Design Foundation"}
    ],
    'Web Developer': [
        {"title": "The Web Developer Bootcamp", "platform": "Udemy (Colt Steele)"},
        {"title": "Full-Stack Web Development with React Specialization", "platform": "Coursera (HKUST)"},
        {"title": "Modern JavaScript (From Beginner to Advanced)", "platform": "Udemy"}
    ]
}

# Helper to check if models are trained and load them
@st.cache_resource
def load_all_models():
    try:
        num_imputer = joblib.load(os.path.join(MODEL_DIR, 'num_imputer.pkl'))
        cat_imputer = joblib.load(os.path.join(MODEL_DIR, 'cat_imputer.pkl'))
        num_cols = joblib.load(os.path.join(MODEL_DIR, 'num_cols.pkl'))
        cat_cols = joblib.load(os.path.join(MODEL_DIR, 'cat_cols.pkl'))
        encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
        target_encoder = joblib.load(os.path.join(MODEL_DIR, 'target_encoder.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        features = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
        
        # Load classical classifiers
        models = {
            'Logistic Regression': joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl')),
            'Random Forest': joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl')),
            'XGBoost': joblib.load(os.path.join(MODEL_DIR, 'xgboost.pkl'))
        }
        
        # Load PyTorch Deep Learning model
        input_dim = len(features)
        num_classes = len(target_encoder.classes_)
        dl_model = CareerMLP(input_dim, num_classes)
        dl_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'deep_learning_mlp.pt')))
        dl_model.eval()
        models['Deep Learning (PyTorch MLP)'] = dl_model
        
        best_info = joblib.load(os.path.join(MODEL_DIR, 'best_model_info.pkl'))
        
        return num_imputer, cat_imputer, num_cols, cat_cols, encoders, target_encoder, scaler, features, models, best_info
    except FileNotFoundError as e:
        return None

def preprocess_single_input(input_df, num_imputer, cat_imputer, num_cols, cat_cols, encoders, scaler, features):
    """
    Applies the pre-fitted preprocessing pipeline objects to a single record.
    """
    df = clean_column_names(input_df)
    
    # Fill standard features
    for col in num_cols:
        if col not in df.columns:
            df[col] = 5.0
            
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
            
    if num_cols:
        df[num_cols] = num_imputer.transform(df[num_cols])
    if cat_cols:
        df[cat_cols] = cat_imputer.transform(df[cat_cols].astype(str))
        
    for col in cat_cols:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else 'Unknown')
            df[col] = le.transform(df[col])
            
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features]
    
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])
        
    return X

def get_skill_suggestions(input_dict):
    """
    Generates personalized advice to optimize student profiles.
    """
    suggestions = []
    if input_dict.get('coding_skills_rating', 5) < 7:
        suggestions.append("🌟 Improve your coding logic and algorithms through online platforms like LeetCode or HackerRank.")
    if input_dict.get('hackathons', 0) < 3:
        suggestions.append("💻 Participate in hackathons and group projects to gain practical software engineering experience.")
    if input_dict.get('public_speaking_points', 5) < 6:
        suggestions.append("🗣️ Enhance your public speaking and communication skills by joining local clubs (e.g. Toastmasters) or delivering workshop presentations.")
    if input_dict.get('logical_quotient_rating', 5) < 7:
        suggestions.append("🧠 Solve more logical puzzles and system analysis problems to boost your analytical aptitude.")
    if input_dict.get('certifications') == 'none':
        suggestions.append("🎓 Acquire standard professional certifications in subjects of interest to validate your skills.")
        
    if not suggestions:
        suggestions.append("🚀 Exceptional profile! Continue learning advanced topics in your domains of interest.")
    return suggestions

# App initialization and custom UI styling
st.set_page_config(page_title="Career Path Prediction & Guidance System", layout="wide", page_icon="🎓")

# Apply custom futuristic glowing dark theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Gradient Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0f172a 0%, #020617 100%);
        color: #f1f5f9;
    }
    
    /* Glassmorphism Title Card */
    .title-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        text-align: center;
    }
    
    /* Customized Form Container */
    .stForm {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 16px !important;
        padding: 30px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    }
    
    /* Glowing Accent Header Styling */
    h1, h2, h3, h4 {
        color: #818cf8 !important;
        font-weight: 700 !important;
    }
    
    /* Premium Styled Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(99, 102, 241, 0.1) !important;
    }
    
    /* Button Hover Micro-animations */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Beautiful Interactive Result Card */
    .result-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.05) 100%);
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    }
    
    /* Elegant Sidebar Info Container */
    .info-container {
        border-left: 3px solid #818cf8;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 12px 16px;
        border-radius: 4px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# App Title banner
st.markdown("""
<div class="title-card">
    <h1>🎓 GLOBAL PROFESSIONAL INTERNSHIP (GPI)</h1>
    <h3 style="color: #cbd5e1 !important; font-weight: 300;">Career Path Prediction & Guidance System</h3>
    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 5px;">Based on deep learning and ensemble ML classifiers evaluating student skills, academic success, and personal interests.</p>
</div>
""", unsafe_allow_html=True)

# Load existing pre-trained model files
pipeline_data = load_all_models()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/isometric/100/diploma.png", width=70)
st.sidebar.markdown("### Navigation Menu")
app_mode = st.sidebar.radio(
    "Go To:", 
    ["Single-Student Career Predictor", "Batch Dataset Prediction (Excel/CSV)", "Dataset Exploratory Analysis (EDA)", "Pipeline Control Center"]
)

if pipeline_data is None:
    st.sidebar.error("⚠️ Models not detected! Please go to the 'Pipeline Control Center' to train the pipeline.")
else:
    num_imputer, cat_imputer, num_cols, cat_cols, encoders, target_encoder, scaler, features, models, best_info = pipeline_data
    st.sidebar.markdown(f"""
    <div class="info-container">
        <span style="font-weight: 600; color: #818cf8;">Active System Info</span><br>
        <span style="font-size: 0.9em; color: #94a3b8;">
            <b>Best Model:</b> {best_info['best_name']}<br>
            <b>Accuracy:</b> {best_info['best_accuracy']:.2%}<br>
            <b>Dataset:</b> PS2_Dataset.csv
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selector for User
    selected_model_name = st.sidebar.selectbox(
        "Choose Inference Model:",
        ["Auto-Ensemble Best"] + list(models.keys())
    )

# ----------------- Navigation Tab 1: Single Student Career Predictor -----------------
if app_mode == "Single-Student Career Predictor":
    if pipeline_data is None:
        st.info("💡 To use this predictor, please run the pipeline first under the **Pipeline Control Center**!")
        st.stop()
        
    st.subheader("🤖 Student Profile Input Form")
    st.write("Fill in the academic ratings, skills, and interests to obtain a personalized career path prediction:")
    
    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📈 Academic & Technical Ratings")
            logical_quotient = st.slider("Logical Quotient (1-10)", 1, 10, 6, help="Problem-solving and reasoning rating.")
            coding_skills = st.slider("Coding Skills (1-10)", 1, 10, 6, help="Standard scripting and coding rating.")
            public_speaking = st.slider("Public Speaking (1-10)", 1, 10, 5, help="Presentation and communication points.")
            hackathons = st.number_input("Hackathons Attended", 0, 20, 1, step=1)
            extra_courses = st.radio("Completed Extra Courses?", ["yes", "no"])
            certifications = st.selectbox(
                "Certifications Completed:",
                ['none', 'python', 'shell programming', 'r programming', 'information security', 'machine learning', 'full stack', 'hadoop', 'app development', 'distro making']
            )

        with col2:
            st.markdown("##### 💭 Preferences & Capabilities")
            self_learning = st.radio("Capable of Self-learning?", ["yes", "no"])
            workshops = st.selectbox(
                "Workshops Attended:",
                ['none', 'testing', 'database security', 'game development', 'data science', 'system designing', 'hacking', 'cloud computing', 'web technologies']
            )
            reading_writing = st.selectbox("Reading & Writing Skill:", ["medium", "poor", "excellent"])
            memory_cap = st.selectbox("Memory Capability Score:", ["medium", "poor", "excellent"])
            interested_subs = st.selectbox(
                "Interested Subject Area:",
                ['programming', 'Management', 'data engineering', 'networks', 'Software Engineering', 'cloud computing', 'parallel computing', 'IOT', 'Computer Architecture', 'hacking']
            )
            interested_career = st.selectbox(
                "Interested Career Area:",
                ['developer', 'testing', 'system developer', 'Business process analyst', 'security', 'cloud computing']
            )

        with col3:
            st.markdown("##### 💼 Personality & Career Vision")
            company_type = st.selectbox(
                "Preferred Settlement Company:",
                ['Product based', 'BPA', 'Cloud Services', 'product development', 'Testing and Maintainance Services', 'SAaS services', 'Web Services', 'Finance', 'Sales and Marketing', 'Service Based']
            )
            senior_inputs = st.radio("Taken career input from elders?", ["yes", "no"])
            book_type = st.selectbox(
                "Interested Book Genres:",
                ['Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art']
            )
            management_tech = st.radio("Management or Technical Focus?", ["Technical", "Management"])
            hard_smart = st.radio("Hard Worker or Smart Worker?", ["smart worker", "hard worker"])
            team_worker = st.radio("Worked in Teams Before?", ["yes", "no"])
            introvert = st.radio("Do you classify as Introvert?", ["no", "yes"])
            
        submit_prediction = st.form_submit_button("🔮 Predict Recommended Career Path")
        
    if submit_prediction:
        # Create input DataFrame
        input_dict = {
            'logical_quotient_rating': logical_quotient,
            'hackathons': hackathons,
            'coding_skills_rating': coding_skills,
            'public_speaking_points': public_speaking,
            'self_learning_capability': self_learning,
            'extra_courses_did': extra_courses,
            'certifications': certifications,
            'workshops': workshops,
            'reading_and_writing_skills': reading_writing,
            'memory_capability_score': memory_cap,
            'interested_subjects': interested_subs,
            'interested_career_area': interested_career,
            'type_of_company_want_to_settle_in': company_type,
            'taken_inputs_from_seniors_or_elders': senior_inputs,
            'interested_type_of_books': book_type,
            'management_or_technical': management_tech,
            'hardsmart_worker': hard_smart,
            'worked_in_teams_ever': team_worker,
            'introvert': introvert
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess input record using fitted preprocessing pipeline
        X_processed = preprocess_single_input(input_df, num_imputer, cat_imputer, num_cols, cat_cols, encoders, scaler, features)
        
        # Determine model to use
        if selected_model_name == "Auto-Ensemble Best":
            model_name = best_info['best_name']
        else:
            model_name = selected_model_name
            
        model = models[model_name]
        
        # Generate Predictions and Probabilities
        if model_name == 'Deep Learning (PyTorch MLP)':
            tensor_input = torch.tensor(X_processed.values, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(tensor_input)
                probas = torch.softmax(outputs, dim=1).numpy()[0]
            top_3_indices = probas.argsort()[-3:][::-1]
            top_3_roles = target_encoder.inverse_transform(top_3_indices)
            top_3_probas = probas[top_3_indices]
            predicted_role = top_3_roles[0]
        else:
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_processed)[0]
                top_3_indices = probas.argsort()[-3:][::-1]
                top_3_roles = target_encoder.inverse_transform(top_3_indices)
                top_3_probas = probas[top_3_indices]
                predicted_role = top_3_roles[0]
            else:
                pred_idx = model.predict(X_processed)
                predicted_role = target_encoder.inverse_transform(pred_idx)[0]
                top_3_roles = [predicted_role]
                top_3_probas = [1.0]
                
        # 3. Display Beautiful Prediction Output
        st.balloons()
        st.markdown(f"""
        <div class="result-card">
            <h3 style="color: #10b981 !important; margin: 0 0 10px 0;">🎉 PREDICTION COMPLETED</h3>
            <p style="font-size: 1.15rem; color: #cbd5e1; margin-bottom: 5px;">Based on the <b>{model_name}</b>, your recommended career path is:</p>
            <h1 style="color: #6366f1 !important; margin: 5px 0 15px 0; font-size: 2.3rem;">{predicted_role}</h1>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Predicted matching probability: <b>{top_3_probas[0]*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Top 3 recommendations inside a sleek column-based bar chart
        st.markdown("### 📊 Top 3 Suggested Matches")
        chart_data = pd.DataFrame({
            'Suggested Career': top_3_roles,
            'Match Confidence (%)': top_3_probas * 100
        })
        
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            fig, ax = plt.subplots(figsize=(7, 3), facecolor='none')
            ax.set_facecolor('none')
            sns.barplot(x='Match Confidence (%)', y='Suggested Career', data=chart_data, hue='Suggested Career', palette='flare', legend=False, ax=ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#475569')
            ax.spines['bottom'].set_color('#475569')
            ax.tick_params(colors='#94a3b8', labelsize=8)
            ax.xaxis.label.set_color('#cbd5e1')
            ax.yaxis.label.set_color('#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig)
        with col_c2:
            st.markdown("##### Role Probabilities Breakdown:")
            for role, prob in zip(top_3_roles, top_3_probas):
                st.write(f"- **{role}**: `{prob*100:.1f}% match`")

        # Display course and action plans
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("### 🎓 Recommended Professional Courses")
            st.write("Enroll in these top-rated programs to gain specialized skills required for this job role:")
            
            courses = COURSE_RECOMMENDATIONS.get(predicted_role, [
                {"title": "Comprehensive Software Engineering Program", "platform": "Coursera"},
                {"title": "Full-Stack Development Specialization", "platform": "edX"}
            ])
            
            for i, c in enumerate(courses, 1):
                st.markdown(f"""
                <div style="background-color: rgba(30, 41, 59, 0.6); padding: 12px 18px; border-radius: 8px; border-left: 4px solid #6366f1; margin-bottom: 10px;">
                    <span style="font-weight: 600; color: #cbd5e1;">{i}. {c['title']}</span><br>
                    <span style="font-size: 0.85em; color: #818cf8;">Offered on: <b>{c['platform']}</b></span>
                </div>
                """, unsafe_allow_html=True)
                
        with col_r:
            st.markdown("### 🚀 Profile Improvement Roadmap")
            st.write("Follow these actions to boost your chances of securing a job in this field:")
            suggestions = get_skill_suggestions(input_dict)
            for sug in suggestions:
                st.info(sug)

# ----------------- Navigation Tab 2: Batch Dataset Prediction -----------------
elif app_mode == "Batch Dataset Prediction (Excel/CSV)":
    if pipeline_data is None:
        st.info("💡 Please compile the pipeline first under the **Pipeline Control Center** tab!")
        st.stop()
        
    st.subheader("📂 Batch Prediction Dashboard")
    st.write("Upload a custom student dataset in **Excel (`.xlsx` or `.xls`)** or **CSV** format to perform automated batch predictions:")
    
    uploaded_file = st.file_uploader("Select student dataset file:", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # 1. Read uploaded file
            filename = uploaded_file.name
            if filename.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
                
            st.success(f"Successfully loaded file `{filename}`! Detected **{df_raw.shape[0]} rows** and **{df_raw.shape[1]} columns**.")
            st.write("##### Sample Preview of Raw Data:")
            st.dataframe(df_raw.head(5))
            
            # 2. Preprocess data dynamically
            st.write("Preprocessing and aligning features...")
            df_cleaned = df_raw.copy()
            
            # Use execute_preprocessing loading mechanism to align column names and values
            X_batch, _ = execute_preprocessing(uploaded_file, is_train=False, fit_scalers=False)
            
            # 3. Model selector for Batch predictions
            batch_model_name = st.selectbox(
                "Choose Active Model for Batch Prediction:",
                list(models.keys())
            )
            
            model = models[batch_model_name]
            
            if st.button("🔮 Execute Batch Predictions"):
                with st.spinner("Processing batch records and calculating matches..."):
                    if batch_model_name == 'Deep Learning (PyTorch MLP)':
                        tensor_input = torch.tensor(X_batch.values, dtype=torch.float32)
                        with torch.no_grad():
                            outputs = model(tensor_input)
                            _, pred_indices = torch.max(outputs, dim=1)
                            pred_indices = pred_indices.numpy()
                        predictions = target_encoder.inverse_transform(pred_indices)
                    else:
                        pred_indices = model.predict(X_batch)
                        predictions = target_encoder.inverse_transform(pred_indices)
                        
                    # Add predictions to dataset
                    df_results = df_raw.copy()
                    df_results['Predicted Suggested Job Role'] = predictions
                    
                    st.success("🎉 Batch predictions computed successfully!")
                    
                    st.write("##### Prediction Summary Preview:")
                    st.dataframe(df_results[['Suggested Job Role', 'Predicted Suggested Job Role'] if 'Suggested Job Role' in df_results.columns else ['Predicted Suggested Job Role']].head(10))
                    
                    # Compute Accuracy if Target column exists
                    if 'Suggested Job Role' in df_raw.columns:
                        correct = (df_results['Suggested Job Role'] == df_results['Predicted Suggested Job Role']).sum()
                        acc = correct / len(df_raw)
                        st.metric(label="Calculated Dataset Match Rate (Accuracy)", value=f"{acc:.2%}")
                        
                    # Class prediction distribution chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    role_counts = pd.Series(predictions).value_counts()
                    sns.barplot(x=role_counts.values, y=role_counts.index, hue=role_counts.index, palette='viridis', legend=False, ax=ax)
                    ax.set_title('Predicted Job Roles Distribution')
                    ax.set_xlabel('Count')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 4. Enable Export to Excel or CSV
                    st.write("##### 📥 Download Predicted Output:")
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        csv_data = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Export as CSV",
                            data=csv_data,
                            file_name=f"predicted_{filename.split('.')[0]}.csv",
                            mime="text/csv"
                        )
                    with col_dl2:
                        # Write to Excel in memory using BytesIO
                        from io import BytesIO
                        output_buffer = BytesIO()
                        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                            df_results.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_data = output_buffer.getvalue()
                        st.download_button(
                            label="Export as Excel Workbook (.xlsx)",
                            data=excel_data,
                            file_name=f"predicted_{filename.split('.')[0]}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
        except Exception as e:
            st.error(f"❌ Error compiling predictions: {e}")

# ----------------- Navigation Tab 3: Dataset Exploratory Analysis (EDA) -----------------
elif app_mode == "Dataset Exploratory Analysis (EDA)":
    st.subheader("📊 Dataset Exploratory Analysis (EDA)")
    
    # Check if visuals are already pre-generated in output
    if os.path.exists(os.path.join(OUTPUT_DIR, 'job_role_distribution.png')):
        st.write("Review the pre-computed EDA reports representing the student profiles dataset:")
        
        # Tabs for visual plots
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🎯 Job Role Distribution", "📈 Feature Correlations", "🎨 Rating Distributions"])
        
        with eda_tab1:
            st.image(os.path.join(OUTPUT_DIR, 'job_role_distribution.png'), caption="Suggested Job Role frequencies inside dataset.", use_container_width=True)
            
        with eda_tab2:
            st.image(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), caption="Correlation heatmap among student ratings and numeric points.", use_container_width=True)
            
        with eda_tab3:
            st.image(os.path.join(OUTPUT_DIR, 'ratings_distribution.png'), caption="Distributions of coding skills, logical quotient, and public speaking.", use_container_width=True)
            
    else:
        st.info("💡 Run the pipeline training in **Pipeline Control Center** or upload a file in **Batch Prediction** to generate exploratory visualization assets!")

# ----------------- Navigation Tab 4: Pipeline Control Center (Retraining) -----------------
elif app_mode == "Pipeline Control Center":
    st.subheader("⚙️ Pipeline Control Center")
    st.write("Use this center to upload your custom student datasets and retrain the Machine Learning & PyTorch Deep Learning pipelines!")
    
    st.markdown("""
    > [!NOTE]
    > Retraining standardizes features, fits data imputers, encodes all category terms (handling out-of-vocabulary terms with 'Unknown' flags), scales variables, and trains **Logistic Regression**, **Random Forest**, **XGBoost**, and **PyTorch Deep Learning** models simultaneously!
    """)
    
    custom_dataset_file = st.file_uploader("Upload custom student dataset (Excel or CSV) for training:", type=['xlsx', 'xls', 'csv'])
    
    if custom_dataset_file is not None:
        filename = custom_dataset_file.name
        temp_data_path = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        # Write file locally
        with open(temp_data_path, 'wb') as f:
            f.write(custom_dataset_file.getvalue())
            
        st.success(f"Loaded training file `{filename}`! File is securely copied to the workspace directory.")
        
        if st.button("🚀 Retrain All Models"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("1. Preprocessing data & standardizing columns...")
                progress_bar.progress(20)
                
                # Check target column existence
                df_temp = pd.read_csv(temp_data_path) if filename.endswith('.csv') else pd.read_excel(temp_data_path)
                df_temp = clean_column_names(df_temp)
                if 'suggested_job_role' not in df_temp.columns:
                    st.error("❌ Preprocessing failed! Uploaded dataset does not contain target column `Suggested Job Role` or `suggested_job_role`!")
                    st.stop()
                
                status_text.text("2. Generating Exploratory Data Analysis (EDA) visuals...")
                progress_bar.progress(40)
                generate_eda_reports(temp_data_path, OUTPUT_DIR)
                
                status_text.text("3. Training ML & PyTorch Deep Learning classifiers (this may take a moment)...")
                progress_bar.progress(60)
                
                # Run the model training pipeline
                metrics = train_and_evaluate(temp_data_path)
                
                progress_bar.progress(90)
                status_text.text("4. Updating active models and saving parameters...")
                
                # Force cache reload by resetting cache
                st.cache_resource.clear()
                
                progress_bar.progress(100)
                st.success("🎉 Pipeline retraining completed successfully!")
                
                # Display metrics comparison
                metrics_df = pd.DataFrame(metrics)
                st.write("##### 📊 Retrained Models Performance Comparison:")
                st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score'], color='#312e81'))
                
                st.info("💡 The Streamlit app has automatically reloaded. All predictions will now use the newly trained best model!")
                
                # Add learning curves image for PyTorch
                if os.path.exists(os.path.join(OUTPUT_DIR, 'dl_training_curves.png')):
                    st.write("##### 📈 PyTorch Deep Learning MLP Training Curves:")
                    st.image(os.path.join(OUTPUT_DIR, 'dl_training_curves.png'))
                    
            except Exception as e:
                st.error(f"❌ Retraining failed with error: {e}")
                
    st.markdown("---")
    st.write("##### Reset system to default dataset:")
    if st.button("🔄 Reset Models to Default Dataset (PS2_Dataset.csv)"):
        with st.spinner("Training on default dataset..."):
            try:
                generate_eda_reports('PS2_Dataset.csv', OUTPUT_DIR)
                train_and_evaluate('PS2_Dataset.csv')
                st.cache_resource.clear()
                st.success("Pipeline reset to default `PS2_Dataset.csv` dataset!")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")
