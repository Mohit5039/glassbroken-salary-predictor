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
from llm import ask_career_bot



# ----------------- Config --------------------
st.set_page_config(
    page_title="Mārga: AI-Powered Career Guide",
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
    st.title("📖 About Mārga: AI-Powered Career Guide")
    st.divider()
    
    st.header("📌 Project Overview")
    st.write(
        "**Mārga** is an AI-powered platform designed to help users with **salary predictions**, **career guidance**, "
        "and **personal upskilling** through interactive, modular interfaces."
    )
    
    st.header("🧠 How It Works")
    st.write(
        "The app uses a trained ML model (XGBoost Regressor) to predict monthly base salary, converts it to annual CTC, "
        "and adjusts for city cost-of-living multipliers. Additionally, it features **CareerBuddy 🤝**, a conversational "
        "career guidance bot powered by **LLaMA**, fine-tuned for this project using prompt engineering to provide "
        "multi-turn, personalized career advice."
    )
    
    st.header("💬 Modular Features")
    st.write(
        "- **PayCheck 💰**: Estimates annual salary, salary ranges, and city-adjusted comparisons with interactive plots.\n"
        "- **CareerBuddy 🤝**: Engages in multi-turn conversations, asks clarifying questions, and provides structured career guidance using LLaMA.\n"
        "- **UpskillGuide 🧠**: Curates learning paths and practice resources to help users upskill step by step."
    )
    
    st.header("📊 Data & Model")
    st.write(
        "Salary prediction uses XGBoost Regressor trained on public Indian salary data. "
        "Preprocessing includes one-hot encoding, scaling, and fuzzy job title matching."
    )
    
    st.header("🏙️ City Adjustments")
    st.write(
        "Salaries are adjusted using static multipliers for major Indian cities. "
        "These are approximations and not learned directly by the model."
    )
    
    st.header("⚠️ Limitations")
    st.write(
        "* CareerBuddy is a first-version LLaMA-based bot; tone and structure will improve over time.\n"
        "* No company-specific salary data.\n"
        "* City adjustments are static.\n"
        "* Predictions may be less accurate for rare job titles.\n"
        "* Gender input is used for analytics only, not salary prediction."
    )
    
    st.header("📈 Future Plans")
    st.write(
        "- Refine CareerBuddy for friendlier, more concise guidance.\n"
        "- Continue daily feature updates and improvements."
        " - Will soon add a feedback feature"
    )
    
    st.header("👥 Credits")
    st.write(
        "Designed and built by **Mohit Singh (Mohit Shekhawat)** as an internship project, "
        "evolving into a modular AI career assistant platform."
    )



def render_chatbot():
    st.title("PayCheck 💰: Know Your Numbers")
    
    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    current_step = len(st.session_state.inputs)

    if current_step < len(FORM_STEPS):
        key, prompt, *options = FORM_STEPS[current_step]
        with st.chat_message("assistant"):
            st.markdown(prompt)

        if options:
            # Dropdown options handling
            dropdown_key = f"{key}_dropdown"
            if dropdown_key not in st.session_state:
                st.session_state[dropdown_key] = None

            select_options = ["Please select..."] + options[0]
            user_response = st.selectbox(
                " ", 
                select_options, 
                key=dropdown_key, 
                label_visibility="collapsed",
                index=0 if st.session_state[dropdown_key] is None else select_options.index(st.session_state[dropdown_key])
            )

            if user_response != "Please select...":
                if st.button("Confirm Selection", key=f"confirm_{key}"):
                    st.session_state.inputs[key] = user_response
                    st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                    st.session_state.messages.append({'role': 'user', 'content': user_response})
                    del st.session_state[dropdown_key]
                    st.rerun()
                else:
                    st.info(f"You selected: **{user_response}**. Click 'Confirm Selection' to proceed.")

        else:
            # Experience field validation
            if key == 'experience':
                user_input = st.chat_input(placeholder="Type your years of experience...")
                if user_input:
                    try:
                        exp_value = float(user_input)
                        age_value = int(st.session_state.inputs.get('age', 0))
                        max_exp = age_value - 18

                        if exp_value < 0:
                            st.error("🤔 **Whoa there!** Negative experience? Keep it positive! 😄")
                            return
                        elif exp_value > max_exp and age_value > 0:
                            st.error(f"🚀 **Hold up!** Max experience with age {age_value} is {max_exp}. Try ≤ {max_exp}")
                            return
                        else:
                            st.session_state.inputs[key] = user_input
                            st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                            st.session_state.messages.append({'role': 'user', 'content': user_input})
                            st.rerun()
                    except ValueError:
                        st.error("🧮 **Oops!** Enter experience in years (e.g., 5, 2.5)")
                        return

            else:
                user_input = st.chat_input(placeholder="Type your answer...")
                if user_input:
                    if key == 'age':
                        try:
                            age_value = int(user_input)
                            if age_value < 18:
                                st.error("👶 **Too young!** Must be at least 18 years old.")
                                return
                        except ValueError:
                            st.error("🎂 **Age should be a whole number!**")
                            return
                    
                    st.session_state.inputs[key] = user_input
                    st.session_state.messages.append({'role': 'assistant', 'content': prompt})
                    st.session_state.messages.append({'role': 'user', 'content': user_input})
                    st.rerun()
    else:
        # All inputs completed
        processed = preprocess_inputs(st.session_state.inputs, all_job_titles)
        if processed[0] is None:
            st.error(processed[1])
            return

        with st.chat_message("assistant"):
            st.markdown("Thanks! Let me calculate your estimated salary... 💰")
        
       
        # Progress animation
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
    
    # Chat summary
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
    # Model output
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
    # City-wise comparison chart
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

    # Detailed city table
    st.subheader("🏙️ Detailed City Comparison")
    display_df = df_cities.copy()
    for col in ['Estimated Salary (Annual)', 'Lower Bound', 'Upper Bound']:
        display_df[col] = display_df[col].apply(lambda x: f"₹ {x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)




    

#----------- Guidance Bot -----------#
def render_career_board():
    """
    Separate Career Guidance Bot (modular).
    Uses ask_career_bot(user_text) from llm.py to answer career queries.
    Conversation history is stored in st.session_state['career_messages'].
    """
    st.title("💡CareerBuddy : Career Guidance Bot")

    # init
    if "career_messages" not in st.session_state:
        st.session_state.career_messages = []

    # Display conversation history
    for msg in st.session_state.career_messages:
        # msg['role'] must be 'user' or 'assistant' (to match your existing pattern)
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Input for new career query (single-line input, multi-turn preserved)
    career_query = st.chat_input(placeholder="Ask about skills, roadmap, interviews, etc...", key="career_query_input")

    if career_query:
        # Append user message and call the LLM
        st.session_state.career_messages.append({'role': 'user', 'content': career_query})
        with st.spinner("Thinking... 🤔"):
            try:
                bot_response = ask_career_bot(career_query)
            except Exception as e:
                bot_response = f"Sorry — I couldn't fetch a response right now. ({e})"

        st.session_state.career_messages.append({'role': 'assistant', 'content': bot_response})
        # rerun to show the updated chat (use st.rerun as you used elsewhere)
        st.rerun()

    # Utility: clear career chat
    if st.button("Clear Career Chat"):
        st.session_state.career_messages = []
        st.rerun()
        
        
def render_upskill_guide():
    st.title("🧠 UpskillGuide: Learn & Grow")
    st.caption("Curated learning paths and practice resources to upskill yourself — one step at a time.")
    st.divider()

    upskill_resources = {
        "Programming": {
            "🎥 Video Resources": {
                "FreeCodeCamp – Full Programming Playlists": "https://www.youtube.com/c/Freecodecamp",
                "Striver – DSA & Competitive Programming": "https://www.youtube.com/@takeUforward",
                "Programming with Mosh – Clean Coding": "https://www.youtube.com/@programmingwithmosh",
                "Tech with Tim – Python Projects": "https://www.youtube.com/@TechWithTim",
                "CodeWithHarry – Beginner Friendly": "https://www.youtube.com/@CodeWithHarry"
            },
            "📚 Reading Sources": {
                "GeeksforGeeks": "https://www.geeksforgeeks.org/",
                "W3Schools": "https://www.w3schools.com/",
                "TutorialsPoint": "https://www.tutorialspoint.com/",
                "RealPython": "https://realpython.com/",
                "CPlusPlus.com": "https://cplusplus.com/"
            },
            "💻 Practice Platforms": {
                "LeetCode": "https://leetcode.com/",
                "HackerRank": "https://www.hackerrank.com/",
                "Codeforces": "https://codeforces.com/",
                "CodeChef": "https://www.codechef.com/",
                "Exercism": "https://exercism.org/"
            }
        },
        "Aptitude & Case Study": {
            "🎥 Video Resources": {
                "IndiaBix Aptitude – YouTube": "https://www.youtube.com/@indiabix",
                "Unacademy Aptitude Series": "https://www.youtube.com/@Unacademy",
                "CareerRide Logical Reasoning": "https://www.youtube.com/@CareerRide",
                "CampusX Placement Series": "https://www.youtube.com/@CampusX-official",
                "SkillSlate – Case Studies": "https://www.youtube.com/@SkillSlate"
            },
            "📚 Reading Sources": {
                "IndiaBix Aptitude & Reasoning": "https://www.indiabix.com/",
                "PrepInsta Resources": "https://prepinsta.com/",
                "GFG Puzzles": "https://www.geeksforgeeks.org/category/puzzles/",
                "CaseInterview.com": "https://www.caseinterview.com/",
                "MindTools Problem Solving": "https://www.mindtools.com/"
            },
            "💻 Practice Platforms": {
                "M4Maths": "https://www.m4maths.com/",
                "TestBook Practice": "https://testbook.com/practice",
                "PrepInsta Quiz Zone": "https://prepinsta.com/quiz/",
                "TalentBattle": "https://talentbattle.in/",
                "BrainBashers": "https://www.brainbashers.com/"
            }
        },
        "Data Analytics": {
            "🎥 Video Resources": {
                "Alex The Analyst": "https://www.youtube.com/@AlexTheAnalyst",
                "Krish Naik – Data Science": "https://www.youtube.com/@krishnaik06",
                "Ken Jee – Career & Projects": "https://www.youtube.com/@KenJee1",
                "Luke Barousse – Data Projects": "https://www.youtube.com/@LukeBarousse",
                "DataCamp Tutorials": "https://www.youtube.com/@DataCamp"
            },
            "📚 Reading Sources": {
                "Kaggle Learn": "https://www.kaggle.com/learn",
                "Analytics Vidhya": "https://www.analyticsvidhya.com/",
                "Towards Data Science": "https://towardsdatascience.com/",
                "Mode SQL Tutorial": "https://mode.com/sql-tutorial/",
                "W3Schools Data Science": "https://www.w3schools.com/python/python_data_science.asp"
            },
            "💻 Practice Platforms": {
                "Kaggle Competitions": "https://www.kaggle.com/competitions",
                "DataCamp Practice": "https://www.datacamp.com/",
                "StrataScratch": "https://www.stratascratch.com/",
                "LeetCode SQL Problems": "https://leetcode.com/problemset/database/",
                "Hackerrank SQL Track": "https://www.hackerrank.com/domains/sql"
            }
        },
        "Development": {
            "🎥 Video Resources": {
                "Traversy Media": "https://www.youtube.com/@TraversyMedia",
                "FreeCodeCamp – Full Stack": "https://www.youtube.com/c/Freecodecamp",
                "Fireship – Modern Concepts": "https://www.youtube.com/@Fireship",
                "Net Ninja": "https://www.youtube.com/@NetNinja",
                "CodeWithHarry – Full Stack Hindi": "https://www.youtube.com/@CodeWithHarry"
            },
            "📚 Reading Sources": {
                "MDN Web Docs": "https://developer.mozilla.org/",
                "DevDocs.io": "https://devdocs.io/",
                "W3Schools Web Dev": "https://www.w3schools.com/",
                "FreeCodeCamp Articles": "https://www.freecodecamp.org/news/",
                "CSS Tricks": "https://css-tricks.com/"
            },
            "💻 Practice Platforms": {
                "Frontend Mentor": "https://www.frontendmentor.io/",
                "CodePen": "https://codepen.io/",
                "GitHub Projects": "https://github.com/trending",
                "JSFiddle": "https://jsfiddle.net/",
                "StackBlitz": "https://stackblitz.com/"
            }
        },
        "Career Growth & Productivity": {
            "🎥 Video Resources": {
                "Ali Abdaal – Productivity": "https://www.youtube.com/@aliabdaal",
                "Thomas Frank – Study & Focus": "https://www.youtube.com/@Thomasfrank",
                "CareerVidz – Interview Prep": "https://www.youtube.com/@CareerVidz",
                "TED Talks – Career Insights": "https://www.youtube.com/@TED",
                "LinkedIn Learning": "https://www.linkedin.com/learning/"
            },
            "📚 Reading Sources": {
                "Harvard Business Review": "https://hbr.org/",
                "Indeed Career Guide": "https://www.indeed.com/career-advice",
                "Medium Career Advice": "https://medium.com/tag/career-advice",
                "MindTools Productivity": "https://www.mindtools.com/",
                "Notion Blog": "https://www.notion.so/blog"
            },
            "💻 Tools & Practice": {
                "LinkedIn Skill Assessments": "https://www.linkedin.com/skill-assessments/",
                "Notion Templates": "https://www.notion.so/templates",
                "Trello": "https://trello.com/",
                "Resume.io": "https://resume.io/",
                "Jobscan Resume Optimizer": "https://www.jobscan.co/"
            }
        }
    }

    tabs = st.tabs(list(upskill_resources.keys()))

    for idx, (category, sections) in enumerate(upskill_resources.items()):
        with tabs[idx]:
            st.markdown(f"### 🚀 {category}")
            st.write("---")

            col1, col2, col3 = st.columns(3)
            columns = [col1, col2, col3]

            for i, (section_name, resources) in enumerate(sections.items()):
                with columns[i % 3]:
                    with st.container(border=True):
                        st.subheader(section_name)
                        for name, link in resources.items():
                            st.markdown(f"- [{name}]({link})", unsafe_allow_html=True)

    st.write("---")
    st.info("💡 More curated paths coming soon. Share your suggestions in the feedback section!")


def render_jobs():
    import requests
    import streamlit as st

    st.title("🔍 Live Job Postings")

    API_KEY = st.secrets["RAPIDAPI_KEY"]

    query = st.text_input("Enter job title (e.g., Data Scientist, Web Developer):")
    location = st.text_input("Enter location (optional):", "India")
    num_pages = st.slider("Number of pages to fetch", 1, 5, 2)

    job_type_filter = st.multiselect(
        "Filter by Job Type",
        options=["Full-time", "Part-time", "Contract", "Internship"],
        default=[]
    )
    company_filter = st.text_input("Filter by Company Name (optional):")

    if st.button("Search Jobs") and query:
        with st.spinner("Fetching live job listings..."):
            url = "https://jsearch.p.rapidapi.com/search"
            headers = {
                "X-RapidAPI-Key": API_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }

            all_jobs = []
            seen_ids = set()

            for page in range(num_pages):
                params = {"query": f"{query} in {location}", "num_pages": 1, "page": page + 1}
                try:
                    response = requests.get(url, headers=headers, params=params)
                    response.raise_for_status()
                    data = response.json()
                    jobs = data.get("data", [])

                    for job in jobs:
                        job_id = job.get("job_id") or job.get("job_apply_link")
                        if job_id not in seen_ids:
                            # Apply filters
                            if job_type_filter and job.get("job_employment_type") not in job_type_filter:
                                continue
                            if company_filter and company_filter.lower() not in job.get("employer_name", "").lower():
                                continue
                            all_jobs.append(job)
                            seen_ids.add(job_id)
                except Exception as e:
                    st.error(f"Error fetching page {page + 1}: {e}")

            if not all_jobs:
                st.warning("No jobs found for your search with applied filters.")
                return

            st.success(f"Found {len(all_jobs)} unique job postings!")

            for i, job in enumerate(all_jobs, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{i}. {job['job_title']}")
                    st.write(f"**Company:** {job['employer_name']}")
                    st.write(f"**Location:** {job['job_city']}, {job['job_country']}")
                    st.write(f"**Type:** {job.get('job_employment_type', 'N/A')}")
                    snippet = job.get("job_description", "")
                    if snippet:
                        st.info(snippet[:300] + ("..." if len(snippet) > 300 else ""))
                    st.markdown(f"[Apply Here 🔗]({job['job_apply_link']})")
                with col2:
                    logo = job.get("employer_logo")
                    if logo:
                        st.image(logo, width=80)
                st.markdown("---")

# ----------------- Main --------------------
# ----------------- Main --------------------
def main():
    global all_job_titles
    model, scaler, expected_columns, all_job_titles = load_assets()

    # ---------------- Initialize Session State ----------------
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'inputs' not in st.session_state:
        st.session_state.inputs = {}
    if 'completed' not in st.session_state:
        st.session_state.completed = False
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False
    if 'show_career' not in st.session_state:
        st.session_state.show_career = False
    if 'show_upskill' not in st.session_state:
        st.session_state.show_upskill = False
    if 'show_jobs' not in st.session_state:     # ✅ Added missing state
        st.session_state.show_jobs = False
    if 'career_messages' not in st.session_state:
        st.session_state.career_messages = []

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("# Mārga: AI-Powered Career Guide")
        st.divider()

        if st.button("About", key="sidebar_about"):
            st.session_state.show_about = True
            st.session_state.show_career = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

        if st.button("PayCheck 💰", key="sidebar_paycheck"):
            st.session_state.show_about = False
            st.session_state.show_career = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.session_state.completed = False
            st.rerun()

        if st.button("CareerBuddy 🤝", key="sidebar_career"):
            st.session_state.show_about = False
            st.session_state.show_career = True
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

        if st.button("UpskillGuide 🧠", key="sidebar_upskill"):
            st.session_state.show_about = False
            st.session_state.show_career = False
            st.session_state.show_upskill = True
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

        if st.button("Live Jobs 💼", key="sidebar_jobs"):
            st.session_state.show_about = False
            st.session_state.show_career = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = True
            st.rerun()

        st.divider()

        # ---------------- Restart Buttons ----------------
        if st.button("Restart PayCheck", key="restart_paycheck"):
            st.session_state.messages = []
            st.session_state.inputs = {}
            st.session_state.completed = False
            st.session_state.show_about = False
            st.session_state.show_career = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

        if st.button("Restart CareerBuddy", key="restart_career"):
            st.session_state.career_messages = []
            st.session_state.show_career = True
            st.session_state.show_about = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

        if st.button("Restart All (Full Reset)", key="restart_all"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.messages = []
            st.session_state.inputs = {}
            st.session_state.completed = False
            st.session_state.show_about = False
            st.session_state.show_career = False
            st.session_state.show_upskill = False
            st.session_state.show_jobs = False   # ✅ added
            st.rerun()

    # ---------------- Page Rendering ----------------
    if st.session_state.show_about:
        render_about_page()
    elif st.session_state.show_career:
        render_career_board()
    elif st.session_state.show_upskill:
        render_upskill_guide()
    elif st.session_state.show_jobs:
        render_jobs()
    elif not st.session_state.completed:
        render_chatbot()
    else:
        render_report(model, scaler, expected_columns, all_job_titles)


if __name__ == "__main__":
    main()
