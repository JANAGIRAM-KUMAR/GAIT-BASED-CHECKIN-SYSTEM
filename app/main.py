import streamlit as st
import pandas as pd
import sys
import os
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_employee

def set_custom_style_with_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    /* Global app background */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Times New Roman', serif !important;
        color: white !important;
    }}

    /* Black sidebar */
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: black !important;
        padding: 20px;
        border-radius: 12px;
    }}

    /* Transparent container */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.0) !important;
    }}

    /* All widget fonts and input styling */
    .stFileUploader, .stTextInput, .stButton > button, .stMarkdown, .stDataFrame {{
        background-color: rgba(255, 255, 255, 0.0) !important;
        color: white !important;
        font-family: 'Times New Roman', serif !important;
    }}

    input, textarea, select {{
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        font-family: 'Times New Roman', serif !important;
    }}

    h1, h2, h3, h4 {{
        color: white !important;
        text-shadow: 1px 1px 2px #000000;
        font-family: 'Times New Roman', serif !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_custom_style_with_background("img/background.jpg")  # ← Use your local image path

st.markdown("# Stark Industries 🏤: Gait-Based Security System")

st.sidebar.header("📌 How to Use")
st.sidebar.markdown("""
1. Upload your **Accelerometer CSV** file.
2. Ensure the following columns exist:
   - `ax (m/s^2)`
   - `ay (m/s^2)`
   - `az (m/s^2)`
   - `aT (m/s^2)`
3. The system will determine if you are an employee.
""")

uploaded_file = st.file_uploader("📂 Upload Accelerometer CSV File", type="csv")
model_path = "models/gait_classifier.pkl"

if uploaded_file is not None:
    try:
        with st.spinner("Processing..."):
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(3), use_container_width=True)

            required_columns = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']
            if all(col in df.columns for col in required_columns):
                employee_id, confidence, model_type = predict_employee(model_path, df)

                st.markdown(f"**🔍 Model Used:** `{model_type}`")
                st.markdown(f"**📈 Confidence:** `{confidence:.2%}`")

                if employee_id:
                    st.success("✅ Access Granted. Employee Recognized.")
                else:
                    st.error("❌ Access Denied. Not recognized.")
            else:
                missing = set(required_columns) - set(df.columns)
                st.error(f"❌ Access Denied. Missing required columns: {', '.join(missing)}")
    except Exception as e:
        st.error(f"🚨 Error: `{str(e)}`")
else:
    st.markdown("#### Upload your CSV file to check employee credentials.")

