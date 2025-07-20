import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
from   sklearn.preprocessing import LabelEncoder
from   rapidfuzz import process, fuzz
import numpy as np
import time

# ----------------- Config --------------------
st.set_page_config(
    page_title="Glassbroken: AI Salary Buddy",
    layout="centered"
    # Removed initial_sidebar_state to ensure sidebar is always visible
)

# ----------------- Load Assets --------------------
model = joblib.load("models/xgboost_final.pkl")
scaler = joblib.load("models/scaler.pkl")
expected_columns = joblib.load("models/columns.pkl")

# ----------------- Constants --------------------
city_salary_ratios = {
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

education_mappings = {
    'btech': 'Bachelor\'s',
    'bachelor of engineering': 'Bachelor\'s',
    'bachelor': 'Bachelor\'s',
    'bsc': 'Bachelor\'s',
    'msc': 'Master\'s',
    'mtech': 'Master\'s',
    'phd': 'PhD',
    'high school': 'High School',
}

job_titles_df = pd.read_csv("data/unique_job_titles.csv")
all_job_titles = job_titles_df['Job Title'].dropna().unique().tolist()

# ----------------- Functions --------------------
def get_closest_job_title(user_input):
    best_match, score, _ = process.extractOne(user_input, all_job_titles, scorer=fuzz.ratio)
    return best_match if score >= 70 else None

def predict_and_adjust(model, scaler, input_df, city):
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    base_monthly = model.predict(scaler.transform(input_df))[0]
    base_annual = base_monthly * 12
    lower = base_annual * 0.72
    upper = base_annual * 1.03
    adjusted = round(base_annual * city_salary_ratios.get(city, 1.0), 2)

    city_salary_df = pd.DataFrame([{
        'City': c,
        'Estimated Salary (Annual)': round(base_annual * r, 2),
        'Lower Bound': round(base_annual * r * 0.72, 2),
        'Upper Bound': round(base_annual * r * 1.03, 2)
    } for c, r in city_salary_ratios.items()]).sort_values(by='Estimated Salary (Annual)', ascending=False)

    return round(base_annual, 2), round(adjusted, 2), round(lower, 2), round(upper, 2), city_salary_df

# ----------------- App State --------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.inputs = {}
    st.session_state.completed = False
    st.session_state.show_about = False

form_steps = [
    ('job', "What's your job title?"),
    ('age', "What's your age?"),
    ('gender', "Select your gender:", ['Male', 'Female', 'Other', 'Prefer not to say']),
    ('education', "What's your highest qualification?"),
    ('experience', "How many years of experience do you have?"),
    ('city', "Select your city:", list(city_salary_ratios.keys()))
]

# ----------------- Sidebar (Always Visible) --------------------
with st.sidebar:
    if st.button("📖 About", key="sidebar_about_btn"):
        st.session_state.show_about = True
        st.rerun()
    
    if st.button("🔄 Restart Chatbot", key="sidebar_restart_btn"):
        # Clear all session state to restart
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ----------------- About Dashboard Page --------------------
if st.session_state.show_about:
    st.title("📖 About Glassbroken: AI Salary Buddy")
    
    st.divider()
    
# ----------------- About Dashboard Page --------------------
if st.session_state.show_about:
    st.title("📖 About Glassbroken: AI Salary Buddy")
    
    # Back button
    if st.button("← Back to Main"):
        st.session_state.show_about = False
        st.rerun()
    
    st.divider()
    
    # Project Overview
    st.header("📌 Project Overview")
    st.write("""
    **Glassbroken: AI Salary Buddy** is a smart, conversational salary prediction app inspired by platforms like Glassdoor. It helps users estimate their annual compensation (CTC) based on job-related inputs such as title, experience, education, and location — all via a friendly chatbot interface. Built with a focus on simplicity and interactivity, it makes data-driven salary insights accessible to everyone, especially freshers and job-switchers.
    """)
    
    # How It Works
    st.header("🧠 How It Works")
    st.write("""
    The app uses a trained machine learning model (XGBoost Regressor) to predict your **monthly base salary**, which is then converted to **annual CTC**. The chatbot collects your inputs one by one and sends them through a cleaned pipeline involving encoding, scaling, and matching your job title with known ones using fuzzy logic. Once the base salary is predicted, the app dynamically adjusts it for your target city using public benchmarks to provide a more realistic CTC range.
    """)
    
    # Data & Model
    st.header("📊 Data & Model")
    st.write("""
    The model was trained on anonymized, publicly sourced Indian salary data. The final choice was **XGBoost Regressor**, selected after comparative testing with other models like Linear Regression and Gradient Boosting. Preprocessing includes one-hot encoding, normalization, and fuzzy job title matching via `rapidfuzz`. The model predicts **monthly salaries**, which we convert to **annual CTC** for reporting.
    """)
    
    # City Adjustments
    st.header("🏙️ City Adjustments")
    st.write("""
    City-wise salary estimates are generated by applying **cost-of-living and compensation scaling factors** to the base prediction. Although the model doesn't learn city directly, we adjust the predicted CTC using static multipliers based on public datasets for major Indian cities like Bangalore, Delhi, and Mumbai.
    """)
    st.info("""
    📝 **Note:** These city-based estimates are approximations and not part of the model training. Real-world salaries can vary significantly by company, role, and negotiation.
    """)
    
    # Limitations
    st.header("⚠️ Limitations")
    st.write("""
    * The model does **not account for company-specific data** or niche tech stacks.
    * City adjustments are **not learned**, but approximated from benchmarks.
    * Predictions may be less accurate for **unusual or rare job titles**.
    * Gender input is **optional** and not used in prediction — it's collected only for potential future analytics.
    * This is a **demo project** and should not be used for actual salary negotiation.
    """)
    
    # Team & Credits
    st.header("👥 Team & Credits")
    st.write("""
    This app was designed and built by **Mohit Singh (aka Mohit Shekhawat)** as part of an internship project titled **Glassbroken: AI-Powered Salary Prediction**. Backend powered by **Scikit-learn**, **XGBoost**, and **RapidFuzz**. Frontend developed in **Streamlit**, with a custom chatbot UI, interactive graphs (Plotly), and conversational flow.
    """)

# ----------------- Chatbot UI --------------------
elif not st.session_state.completed:
    st.title("💬 Glassbroken Salary Chatbot")

    # Display past messages (only in chatbot mode)
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Get next question
    current_step = len(st.session_state.inputs)
    if current_step < len(form_steps):
        key, prompt, *options = form_steps[current_step]

        with st.chat_message("assistant"):
            st.markdown(prompt)

        if options:
            # For dropdown questions (gender and city)
            user_response = st.selectbox(
                " ", 
                options[0], 
                key=key, 
                label_visibility="collapsed",
                index=None,  # No default selection
                placeholder=f"Choose {key}..."
            )
            
            if user_response is not None:
                st.session_state.inputs[key] = user_response
                st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                st.session_state.messages.append({'role': 'user', 'content': user_response})
                st.rerun()
        else:
            # For text input questions
            user_input = st.chat_input(placeholder="Type your answer...")
            if user_input:
                st.session_state.inputs[key] = user_input
                st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                st.session_state.messages.append({'role': 'user', 'content': user_input})
                st.rerun()
    else:
        with st.chat_message("assistant"):
            st.markdown("Thanks! Let me calculate your estimated salary...")
        
        # Progress indicator
        progress_bar = st.progress(0)
        for i in range(101):
            progress_bar.progress(i)
            time.sleep(0.01)
        
        st.session_state.completed = True
        st.rerun()

# ----------------- Clean Dashboard (No Chat History) --------------------
if st.session_state.completed:
    data = st.session_state.inputs.copy()
    matched_job = get_closest_job_title(data['job']) or data['job']
    data['job_grouped'] = matched_job
    data['Education Level'] = education_mappings.get(data['education'].lower(), 'Others')

    # Prepare model input
    model_input = pd.DataFrame([{
        'Age': int(data['age']),
        'Gender': data['gender'],
        'Education Level': data['Education Level'],
        'Years of Experience': float(data['experience']),
        'job_grouped': data['job_grouped']
    }])

    # Encode categorical variables
    for col in model_input.select_dtypes('object'):
        le = LabelEncoder()
        model_input[col] = le.fit_transform(model_input[col])

    # Get predictions
    base, city_salary, low, high, df_cities = predict_and_adjust(model, scaler, model_input, data['city'])

    # Clean Dashboard Layout
    st.title("📊 Salary Prediction Report")
    
    # Chat Summary Section
    st.subheader("🧾 Chat Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Matched Job Title:** {matched_job}")
        st.write(f"**Age:** {data['age']} years")
        st.write(f"**Gender:** {data['gender']}")
    
    with col2:
        st.write(f"**Education:** {data['Education Level']}")
        st.write(f"**Experience:** {data['experience']} years")
        st.write(f"**Location:** {data['city']}")

    st.divider()

    # Salary Estimates Section
    st.subheader("💰 Estimated Annual CTC (Model Output)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Base Annual Salary",
            value=f"₹ {base:,.0f}",
            help="Salary estimate from ML model"
        )
    with col2:
        st.metric(
            label="Salary Range",
            value=f"₹ {low:,.0f} - ₹ {high:,.0f}",
            help="Lower and upper bounds of estimate"
        )

    st.subheader(f"📍 City-Adjusted Salary for {data['city']}")
    st.metric(
        label=f"Adjusted for {data['city']}",
        value=f"₹ {city_salary:,.0f}",
        delta=f"₹ {city_salary - base:,.0f}",
        help="Salary adjusted for cost of living in selected city"
    )

    st.info(
        "💡 **Note:** City-based salary estimates are derived from public cost-of-living and compensation benchmarks "
        "across 10 major Indian cities. These are **not part of the trained ML model** and should be treated as rough approximations."
    )

    st.divider()

    # Interactive Graph Section
    st.subheader("📈 City-wise Salary Comparison")
    
    fig = go.Figure()
    
    # Add city salary line
    fig.add_trace(go.Scatter(
        x=df_cities['City'],
        y=df_cities['Estimated Salary (Annual)'],
        mode='lines+markers',
        name='City-adjusted CTC',
        marker=dict(size=8, color='lightblue'),
        line=dict(width=2, color='lightblue')
    ))
    
    # Highlight user's city
    fig.add_trace(go.Scatter(
        x=[data['city']],
        y=[city_salary],
        mode='markers',
        name=f'Your City ({data["city"]})',
        marker=dict(color='blue', size=16, symbol='circle', line=dict(width=3, color='darkblue'))
    ))
    
    # Add base model prediction
    fig.add_trace(go.Scatter(
        x=["Base Model"],
        y=[base],
        mode='markers',
        name='Base CTC (Model)',
        marker=dict(color='red', size=14, symbol='diamond', line=dict(width=2, color='darkred'))
    ))

    fig.update_layout(
        title="Annual CTC Estimates Across Cities",
        xaxis_title="Cities / Model Output",
        yaxis_title="Annual Salary (₹)",
        showlegend=True,
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # City Comparison Table
    st.subheader("🏙️ Detailed City Comparison")
    
    # Format the dataframe for better display
    display_df = df_cities.copy()
    display_df['Estimated Salary (Annual)'] = display_df['Estimated Salary (Annual)'].apply(lambda x: f"₹ {x:,.0f}")
    display_df['Lower Bound'] = display_df['Lower Bound'].apply(lambda x: f"₹ {x:,.0f}")
    display_df['Upper Bound'] = display_df['Upper Bound'].apply(lambda x: f"₹ {x:,.0f}")
    
    st.dataframe(
        display_df, 
        use_container_width=True,
        hide_index=True,
        column_config={
            "City": st.column_config.TextColumn("City", width="medium"),
            "Estimated Salary (Annual)": st.column_config.TextColumn("Annual CTC", width="medium"),
            "Lower Bound": st.column_config.TextColumn("Lower Bound", width="medium"),
            "Upper Bound": st.column_config.TextColumn("Upper Bound", width="medium")
        }
    )
