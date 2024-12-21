import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained classifier model
clf = joblib.load('employee_churn_voting_model.joblib')  # Replace with your actual model filename

# Application title and description
st.title("📊 Employee Churn Analytics System")
st.write("This app predicts whether an employee is likely to leave the company based on key performance indicators.")

# Section 1: Employee Information
st.header("🧑‍💼 Employee Information")
staf_name = st.text_input("Enter your staff name:")

# Section 2: Performance Metrics
st.header("📈 Performance Metrics")
satisfaction_level = st.slider("Employee Satisfaction (0-1):", min_value=0.0, max_value=1.0, value=0.5, help="How satisfied is the employee?")
last_evaluation = st.slider("Evaluation Grade (0-1):", min_value=0.0, max_value=1.0, value=0.5, help="The last evaluation score given to the employee.")

# Section 3: Company Metrics
st.header("🏢 Company Metrics")
number_project = st.number_input("Project Involvement:", min_value=1, max_value=50, value=5)
average_montly_hours = st.number_input("Average Monthly Hours:", min_value=0, max_value=744, value=160)
time_spend_company = st.number_input("Time Spent at Company (Years):", min_value=1, max_value=50, value=3)
Work_accident = st.selectbox("Work Accident (0 for No, 1 for Yes):", options=[0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years (0 for No, 1 for Yes):", options=[0, 1])

# Organize inputs into a NumPy array
new_input = np.array([[satisfaction_level, last_evaluation, number_project, average_montly_hours,
                       time_spend_company, Work_accident, promotion_last_5years]])

# Predict and display results
if st.button("🚀 Predict"):
    predictNew = clf.predict(new_input)
    
    st.subheader("📝 Classification Result:")
    if predictNew == [1]:
        st.error(f"❗ Predicted Result: Sorry, {staf_name} tends to leave the company.")
    else:
        st.success(f"✅ Predicted Result: Good news! Most likely {staf_name} will stay.")

# Footer
st.write("⚙️ *This prediction is based on machine learning insights and model evaluation metrics.*")
