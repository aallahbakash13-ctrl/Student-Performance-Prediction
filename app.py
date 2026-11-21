
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# DARK THEME CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1120, #111827, #17132b);
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #21183d);
    border-right: 1px solid #343052;
}

h1, h2, h3 {
    color: #e0e7ff !important;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

.hero {
    background: linear-gradient(120deg, #312e81, #6d28d9, #0891b2);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 30px rgba(109, 40, 217, 0.25);
}

.hero h1 {
    color: white !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.metric-card {
    background: linear-gradient(145deg, #1e293b, #252044);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #51418a;
    text-align: center;
}

.metric-label {
    color: #a5b4fc;
    font-size: 14px;
}

.metric-value {
    color: #67e8f9;
    font-size: 27px;
    font-weight: bold;
}

.result-card {
    background: linear-gradient(120deg, #312e81, #6d28d9);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #a78bfa;
    margin: 15px 0;
}

.result-card h2 {
    color: white !important;
}

.result-score {
    font-size: 42px;
    color: #67e8f9;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #0891b2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #6d28d9, #0e7490);
    color: white;
    border: 1px solid #67e8f9;
}

div[data-testid="stForm"] {
    background: #151d30;
    border: 1px solid #39345e;
    padding: 18px;
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "student_performance_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "datasets", "StudentPerformanceFactors.csv.csv"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check the models folder.")
    st.stop()

model = load_model(MODEL_PATH)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🎓 Student Performance Prediction</h1>
    <p>AI-powered student exam score prediction and analytics dashboard</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR INPUT FORM
# --------------------------------------------------

st.sidebar.title("🎯 Student Details")
st.sidebar.caption("Enter student information below")

with st.sidebar.form("student_prediction_form"):

    st.subheader("📚 Academic Details")

    hours = st.number_input(
        "Hours Studied",
        min_value=0,
        max_value=24,
        value=5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=85
    )

    previous_scores = st.number_input(
        "Previous Scores",
        min_value=0,
        max_value=100,
        value=70
    )

    tutoring = st.number_input(
        "Tutoring Sessions",
        min_value=0,
        max_value=20,
        value=1
    )

    st.subheader("🏠 Student Background")

    parental_involvement = st.selectbox(
        "Parental Involvement",
        ["Low", "Medium", "High"]
    )

    resources = st.selectbox(
        "Access to Resources",
        ["Low", "Medium", "High"]
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    sleep = st.number_input(
        "Sleep Hours",
        min_value=0,
        max_value=24,
        value=7
    )

    motivation = st.selectbox(
        "Motivation Level",
        ["Low", "Medium", "High"]
    )

    internet = st.selectbox(
        "Internet Access",
        ["Yes", "No"]
    )

    family_income = st.selectbox(
        "Family Income",
        ["Low", "Medium", "High"]
    )

    teacher_quality = st.selectbox(
        "Teacher Quality",
        ["Low", "Medium", "High"]
    )

    school_type = st.selectbox(
        "School Type",
        ["Public", "Private"]
    )

    peer_influence = st.selectbox(
        "Peer Influence",
        ["Positive", "Neutral", "Negative"]
    )

    physical_activity = st.number_input(
        "Physical Activity (hours)",
        min_value=0,
        max_value=24,
        value=3
    )

    learning_disabilities = st.selectbox(
        "Learning Disabilities",
        ["Yes", "No"]
    )

    parental_education = st.selectbox(
        "Parental Education Level",
        ["High School", "College", "Postgraduate"]
    )

    distance = st.selectbox(
        "Distance from Home",
        ["Near", "Moderate", "Far"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    submit = st.form_submit_button(
        "🚀 Predict Exam Score"
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submit:

    input_data = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet,
        "Tutoring_Sessions": tutoring,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance,
        "Gender": gender
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.session_state.prediction_history.append({
            "Hours Studied": hours,
            "Attendance": attendance,
            "Previous Scores": previous_scores,
            "Predicted Exam Score": round(float(prediction), 2)
        })

        st.markdown(f"""
        <div class="result-card">
            <h2>✨ Prediction Result</h2>
            <div class="result-score">{prediction:.2f}</div>
            <p>Estimated Exam Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an estimated score from the trained ML model, "
            "not a guaranteed exam result."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------------------------------
# DASHBOARD DATA
# --------------------------------------------------

st.divider()
st.header("📊 Student Performance Dashboard")

if os.path.exists(DATA_PATH):

    df = pd.read_csv(DATA_PATH)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👨‍🎓 Total Students</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        average_score = df["Exam_Score"].mean()

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Average Exam Score</div>
            <div class="metric-value">{average_score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔮 Predictions Made</div>
            <div class="metric-value">{len(st.session_state.prediction_history)}</div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------
    # CHART 1: EXAM SCORE DISTRIBUTION
    # --------------------------------------------------

    st.subheader("📊 Exam Score Distribution")

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor("#151d30")
    ax1.set_facecolor("#151d30")

    ax1.hist(
        df["Exam_Score"],
        bins=25,
        color="#8b5cf6",
        edgecolor="#c4b5fd",
        alpha=0.9
    )

    ax1.set_xlabel("Exam Score", color="white")
    ax1.set_ylabel("Number of Students", color="white")
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#64748b")
    ax1.spines["left"].set_color("#64748b")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    st.pyplot(fig1)
    plt.close(fig1)

    # --------------------------------------------------
    # CHART 2: STUDY HOURS VS EXAM SCORE
    # --------------------------------------------------

    st.subheader("📚 Study Hours vs Exam Score")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#151d30")
    ax2.set_facecolor("#151d30")

    ax2.scatter(
        df["Hours_Studied"],
        df["Exam_Score"],
        color="#22d3ee",
        alpha=0.5
    )

    ax2.set_xlabel("Hours Studied", color="white")
    ax2.set_ylabel("Exam Score", color="white")
    ax2.tick_params(colors="white")
    ax2.spines["bottom"].set_color("#64748b")
    ax2.spines["left"].set_color("#64748b")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    st.pyplot(fig2)
    plt.close(fig2)

    # --------------------------------------------------
    # CHART 3: ATTENDANCE VS EXAM SCORE
    # --------------------------------------------------

    st.subheader("📝 Attendance vs Exam Score")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    fig3.patch.set_facecolor("#151d30")
    ax3.set_facecolor("#151d30")

    ax3.scatter(
        df["Attendance"],
        df["Exam_Score"],
        color="#34d399",
        alpha=0.5
    )

    ax3.set_xlabel("Attendance (%)", color="white")
    ax3.set_ylabel("Exam Score", color="white")
    ax3.tick_params(colors="white")
    ax3.spines["bottom"].set_color("#64748b")
    ax3.spines["left"].set_color("#64748b")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    st.pyplot(fig3)
    plt.close(fig3)

else:
    st.warning(
        "Dataset not found. Please check the datasets folder "
        "and the CSV filename."
    )

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.divider()
st.header("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    average_prediction = history_df[
        "Predicted Exam Score"
    ].mean()

    st.metric(
        "Average Predicted Score",
        f"{average_prediction:.2f}"
    )

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info(
        "No predictions available yet. Make your first prediction!"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:15px;">
    <p>🎓 Student Performance Prediction | Machine Learning Project</p>
    <p>Built with Python, Scikit-learn and Streamlit</p>
</div>
""", unsafe_allow_html=True)
