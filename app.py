import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

# Page setup
st.set_page_config(page_title="Salary Estimation App", layout="wide")
st.title("💼 Salary Estimation App")
st.markdown("#### Predict your expected salary based on role and experience!")

# Cute animated gif
st.image("https://media.giphy.com/media/3o6gDWzmAzrpi5DQU8/giphy.gif", caption="Let’s predict!", use_container_width=True)

st.divider()

# Job roles list (should match what was used during model training!)
job_roles = ["Data Scientist", "Software Engineer", "HR", "Product Manager", "Marketing Executive", "Business Analyst"]

# Layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    selected_role = st.selectbox("🧑‍💼 Job Role", job_roles)

with col2:
    experience_years = st.slider("👩‍💻 Total Experience (Years)", 0, 30, 2)

with col3:
    satisfaction_level = st.slider("😊 Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01, value=0.7)

average_monthly_hours = st.slider("📅 Avg Monthly Hours", min_value=120, max_value=310, step=1, value=160)

# Load model, scaler, and encoder
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")
role_encoder = joblib.load("role_encoder.pkl")  # You must have saved this during training

# Encode job role
encoded_role = role_encoder.transform([selected_role])[0]

# Feature vector
X = [experience_years, encoded_role, satisfaction_level, average_monthly_hours]

# Predict button
predict_button = st.button("🚀 Predict Salary")
st.divider()

if predict_button:
    st.balloons()

    X_array = scaler.transform([np.array(X)])
    prediction = model.predict(X_array)

    st.success(f"🎯 Predicted Salary for a **{selected_role}** with {experience_years} years experience: **₹ {prediction[0]:,.2f}**")

    # Visualize input
    df_input = pd.DataFrame({
        "Feature": ["Experience (Years)", "Job Role (Encoded)", "Satisfaction", "Avg Monthly Hours"],
        "Value": X
    })

    fig = px.bar(df_input, x="Feature", y="Value", color="Feature",
                 title="📊 Your Input Profile", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Fill the details and press the **Predict Salary** button.")
