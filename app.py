import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process, fuzz
import numpy as np
import time
from xgboost import XGBRegressor
from xgboost.core import XGBoostError

# ----------------- Config --------------------
st.set_page_config(
    page_title="Glassbroken: AI Salary Buddy",
    layout="centered"
)

# ----------------- Constants --------------------
CITY_SALARY_RATIOS = {
    'Bangalore': 1.15,
    'Mumbai': 1.10,
    'Delhi': 1.05,
    'Hyderabad': 1.00,
    'Chennai': 0.95,
    'Pune': 0.98,
    'Gurgaon': 1.07,
    'Noida': 1.02,
    'Kolkata': 0.90,
    'Jaipur': 0.88
}

EDUCATION_MAPPINGS = {
    'btech': "Bachelor's",
    'bachelor of engineering': "Bachelor's",
    'bachelor': "Bachelor's",
    'bsc': "Bachelor's",
    'msc': "Master's",
    'mtech': "Master's",
    'phd': "PhD",
    'high school': "High School",
}

FORM_STEPS = [
    ('job', "What's your job title?"),
    ('age', "What's your age?"),
    ('gender', "Select your gender:", ['Male', 'Female', 'Other', 'Prefer not to say']),
    ('education', "What's your highest qualification?"),
    ('experience', "How many years of experience do you have?"),
    ('city', "Select your city:", list(CITY_SALARY_RATIOS.keys()))
]

# ----------------- Load Assets --------------------
def load_assets():
    model = XGBRegressor()
    try:
        model.load_model("models/xgboost_final.json")
    except XGBoostError:
        model = joblib.load("models/xgboost_final.pkl")

    scaler = joblib.load("models/scaler.pkl")
    expected_columns = joblib.load("models/columns.pkl")
    job_titles_df = pd.read_csv("data/unique_job_titles.csv")
    all_job_titles = job_titles_df['Job Title'].dropna().unique().tolist()

    return model, scaler, expected_columns, all_job_titles

# ----------------- Helper Functions --------------------
def get_closest_job_title(user_input, all_job_titles):
    best_match, score, _ = process.extractOne(user_input, all_job_titles, scorer=fuzz.ratio)
    return best_match if score >= 70 else None

def preprocess_inputs(inputs, all_job_titles):
    matched_job = get_closest_job_title(inputs['job'], all_job_titles) or inputs['job']
    edu_level = EDUCATION_MAPPINGS.get(inputs['education'].lower(), 'Others')

    try:
        age = int(inputs['age'])
        exp = float(inputs['experience'])
    except ValueError:
        return None, "Age or Experience is not a valid number."

    # ----------------- Edge Case Handling ----------------
    if exp < 0:
        return None, "Experience cannot be negative."
    if age < 18:
        return None, "Age must be at least 18."
    if exp > (age - 18):
        return None, f"Experience cannot exceed {age - 18} years based on your age."

    df = pd.DataFrame([{
        'Age': age,
        'Gender': inputs['gender'],
        'Education Level': edu_level,
        'Years of Experience': exp,
        'job_grouped': matched_job
    }])

    for col in df.select_dtypes('object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df, matched_job, edu_level

def predict_and_adjust(model, scaler, expected_columns, input_df, city):
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    base_monthly = model.predict(scaler.transform(input_df))[0]
    base_annual = base_monthly * 12
    lower = base_annual * 0.72
    upper = base_annual * 1.03
    adjusted = round(base_annual * CITY_SALARY_RATIOS.get(city, 1.0), 2)

    city_salary_df = pd.DataFrame([{
        'City': c,
        'Estimated Salary (Annual)': round(base_annual * r, 2),
        'Lower Bound': round(base_annual * r * 0.72, 2),
        'Upper Bound': round(base_annual * r * 1.03, 2)
    } for c, r in CITY_SALARY_RATIOS.items()]).sort_values(by='Estimated Salary (Annual)', ascending=False)

    return round(base_annual, 2), round(adjusted, 2), round(lower, 2), round(upper, 2), city_salary_df

# ----------------- Pages --------------------
def render_about_page():
    st.title("📖 About Glassbroken: AI Salary Buddy")
    st.divider()
    st.header("📌 Project Overview")
    st.write("**Glassbroken: AI Salary Buddy** helps users estimate annual compensation (CTC) "
             "based on job-related inputs like title, experience, education, and city "
             "through a friendly chatbot interface.")
    st.header("🧠 How It Works")
    st.write("The app uses a trained ML model (XGBoost Regressor) to predict monthly base salary, "
             "converts it to annual CTC, and adjusts for city cost-of-living multipliers.")
    st.header("📊 Data & Model")
    st.write("Model: XGBoost Regressor, trained on public Indian salary data. "
             "Preprocessing: one-hot encoding, scaling, fuzzy job title matching.")
    st.header("🏙️ City Adjustments")
    st.write("Salaries are adjusted with static multipliers for major Indian cities. "
             "These are approximations and not learned directly by the model.")
    st.header("⚠️ Limitations")
    st.write("* No company-specific data\n* City adjustments are static\n* Predictions may be less accurate for rare job titles\n* Gender input not used in model (only for analytics)")
    st.header("👥 Credits")
    st.write("Designed and built by **Mohit Singh (Mohit Shekhawat)** as an internship project.")

def render_chatbot():
    st.title("💬 Glassbroken Salary Chatbot")
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    current_step = len(st.session_state.inputs)
    if current_step < len(FORM_STEPS):
        key, prompt, *options = FORM_STEPS[current_step]
        with st.chat_message("assistant"):
            st.markdown(prompt)

        if options:
            # Initialize the dropdown state if not exists
            dropdown_key = f"{key}_dropdown"
            if dropdown_key not in st.session_state:
                st.session_state[dropdown_key] = None
            
            # Create selectbox with a placeholder option
            select_options = ["Please select..."] + options[0]
            user_response = st.selectbox(
                " ", 
                select_options, 
                key=dropdown_key, 
                label_visibility="collapsed",
                index=0 if st.session_state[dropdown_key] is None else select_options.index(st.session_state[dropdown_key])
            )
            
            # Only proceed if user made a valid selection
            if user_response != "Please select...":
                # Add a button to confirm the selection
                if st.button("Confirm Selection", key=f"confirm_{key}"):
                    st.session_state.inputs[key] = user_response
                    st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                    st.session_state.messages.append({'role': 'user', 'content': user_response})
                    # Clean up the dropdown state
                    if dropdown_key in st.session_state:
                        del st.session_state[dropdown_key]
                    st.rerun()
                else:
                    st.info(f"You selected: **{user_response}**. Click 'Confirm Selection' to proceed.")
        else:
            # Special handling for experience field with real-time validation
            if key == 'experience':
                user_input = st.chat_input(placeholder="Type your years of experience...")
                if user_input:
                    # Real-time validation for experience
                    try:
                        exp_value = float(user_input)
                        age_value = int(st.session_state.inputs.get('age', 0))
                        max_exp = age_value - 18
                        
                        if exp_value < 0:
                            st.error("🤔 **Whoa there, time traveler!** Negative experience? Unless you've been working in a parallel universe, let's keep it positive! 😄")
                            return
                        elif exp_value > max_exp and age_value > 0:
                            st.error(f"🚀 **Hold up, superhuman!** With {age_value} years of age, you could have max {max_exp} years of experience (assuming you started working at 18). Did you start working in the womb? 😅 Try a number ≤ {max_exp}")
                            return
                        else:
                            # Valid input, proceed
                            st.session_state.inputs[key] = user_input
                            st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                            st.session_state.messages.append({'role': 'user', 'content': user_input})
                            st.rerun()
                    except ValueError:
                        st.error("🧮 **Oops!** That doesn't look like a number. Please enter your experience in years (e.g., 5, 2.5, etc.)")
                        return
            else:
                user_input = st.chat_input(placeholder="Type your answer...")
                if user_input:
                    # Special validation for age
                    if key == 'age':
                        try:
                            age_value = int(user_input)
                            if age_value < 18:
                                st.error("👶 **Too young for the workforce!** You need to be at least 18 years old. Come back when you're all grown up! 😊")
                                return
                        except ValueError:
                            st.error("🎂 **Age should be a whole number!** Please enter your age in years (e.g., 25, 30, etc.)")
                            return
                    
                    st.session_state.inputs[key] = user_input
                    st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                    st.session_state.messages.append({'role': 'user', 'content': user_input})
                    st.rerun()
    else:
        # Preprocess inputs to check for edge case before showing progress
        processed = preprocess_inputs(st.session_state.inputs, all_job_titles)
        if processed[0] is None:
            st.error(processed[1])
            return
        with st.chat_message("assistant"):
            st.markdown("Thanks! Let me calculate your estimated salary...")
        progress_bar = st.progress(0)
        for i in range(101):
            progress_bar.progress(i)
            time.sleep(0.01)
        st.session_state.completed = True
        st.rerun()

def render_report(model, scaler, expected_columns, all_job_titles):
    data = st.session_state.inputs.copy()
    processed = preprocess_inputs(data, all_job_titles)
    if processed[0] is None:
        st.error(processed[1])
        return

    model_input, matched_job, edu_level = processed
    base, city_salary, low, high, df_cities = predict_and_adjust(model, scaler, expected_columns, model_input, data['city'])

    st.title("📊 Salary Prediction Report")
    st.subheader("🧾 Chat Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Matched Job Title:** {matched_job}")
        st.write(f"**Age:** {data['age']} years")
        st.write(f"**Gender:** {data['gender']}")
    with col2:
        st.write(f"**Education:** {edu_level}")
        st.write(f"**Experience:** {data['experience']} years")
        st.write(f"**Location:** {data['city']}")

    st.divider()
    st.subheader("💰 Estimated Annual CTC (Model Output)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Annual Salary", f"₹ {base:,.0f}", help="From ML model")
    with col2:
        st.metric("Salary Range", f"₹ {low:,.0f} - ₹ {high:,.0f}")

    st.subheader(f"📍 City-Adjusted Salary for {data['city']}")
    st.metric(
        label=f"Adjusted for {data['city']}",
        value=f"₹ {city_salary:,.0f}",
        delta=f"₹ {city_salary - base:,.0f}",
        help="Salary adjusted for cost of living"
    )

    st.divider()
    st.subheader("📈 City-wise Salary Comparison")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_cities['City'], y=df_cities['Estimated Salary (Annual)'],
        mode='lines+markers', name='City-adjusted CTC',
        marker=dict(size=8, color='lightblue'), line=dict(width=2, color='lightblue')
    ))
    fig.add_trace(go.Scatter(
        x=[data['city']], y=[city_salary],
        mode='markers', name=f'Your City ({data["city"]})',
        marker=dict(color='blue', size=16, symbol='circle', line=dict(width=3, color='darkblue'))
    ))
    fig.add_trace(go.Scatter(
        x=["Base Model"], y=[base],
        mode='markers', name='Base CTC (Model)',
        marker=dict(color='red', size=14, symbol='diamond', line=dict(width=2, color='darkred'))
    ))
    fig.update_layout(
        title="Annual CTC Estimates Across Cities",
        xaxis_title="Cities / Model Output",
        yaxis_title="Annual Salary (₹)",
        height=500, hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏙️ Detailed City Comparison")
    display_df = df_cities.copy()
    for col in ['Estimated Salary (Annual)', 'Lower Bound', 'Upper Bound']:
        display_df[col] = display_df[col].apply(lambda x: f"₹ {x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ----------------- Main --------------------
def main():
    global all_job_titles
    model, scaler, expected_columns, all_job_titles = load_assets()

    # ---------------- Initialize Session State ----------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.inputs = {}
        st.session_state.completed = False
        st.session_state.show_about = False

    # ---------------- Sidebar ----------------
    with st.sidebar:
        if st.button("About"):
            st.session_state.show_about = True
            st.rerun()
        if st.button("Restart Chatbot"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.messages = []
            st.session_state.inputs = {}
            st.session_state.completed = False
            st.session_state.show_about = False
            st.rerun()

    # ---------------- Page Rendering ----------------
    if st.session_state.show_about:
        render_about_page()
    elif not st.session_state.completed:
        render_chatbot()
    else:
        render_report(model, scaler, expected_columns, all_job_titles)

if __name__ == "__main__":
    main()