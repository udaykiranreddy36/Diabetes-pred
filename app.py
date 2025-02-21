import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from fpdf import FPDF

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to load files with error handling
def load_file(file_path):
    if not os.path.exists(file_path):
        st.error(f"Error: File '{file_path}' not found. Ensure the file is in the correct directory.")
        st.stop()
    return joblib.load(file_path)

# Load saved model and scaler
try:
    model = load_file('best_model.pkl')
    scaler = load_file('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Feature names (must match the ones used during fitting)
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# App title
st.title("Explainable AI Medical Diagnosis for Diabetes")

# Input form for user data
st.sidebar.header("Input Features")
user_input = {}

# Organize input fields vertically
st.sidebar.subheader("Basic Health Metrics")
user_input['Glucose'] = st.sidebar.slider(
    "Glucose (mg/dL)", 0, 200, 100, 1,
    help="Normal range: 70–140 mg/dL. Higher values may indicate diabetes."
)
user_input['BloodPressure'] = st.sidebar.slider(
    "Blood Pressure (mm Hg)", 0, 140, 80, 1,
    help="Normal range: 60–80 mm Hg. High blood pressure increases diabetes risk."
)
user_input['BMI'] = st.sidebar.slider(
    "BMI (kg/m²)", 0.0, 60.0, 25.0, 0.1,
    help="Normal range: 18.5–24.9 kg/m². Higher BMI increases diabetes risk."
)

st.sidebar.subheader("Advanced Health Metrics")
user_input['SkinThickness'] = st.sidebar.slider(
    "Skin Thickness (mm)", 0, 60, 20, 1,
    help="Normal range: 10–40 mm. Higher values may indicate higher body fat."
)
user_input['Insulin'] = st.sidebar.slider(
    "Insulin (µIU/mL)", 0, 300, 50, 1,
    help="Normal range: 2.6–24.9 µIU/mL. Higher values may indicate insulin resistance."
)

st.sidebar.subheader("Personal and Genetic Factors")
user_input['Pregnancies'] = st.sidebar.slider(
    "Pregnancies", 0, 20, 0, 1,
    help="Normal range: 0–5. Higher pregnancies may increase gestational diabetes risk."
)
user_input['DiabetesPedigreeFunction'] = st.sidebar.slider(
    "Diabetes Pedigree Function", 0.0, 2.0, 0.5, 0.01,
    help="Normal range: 0.0–2.0. Higher values indicate stronger family history of diabetes."
)
user_input['Age'] = st.sidebar.slider(
    "Age (years)", 0, 100, 30, 1,
    help="Diabetes risk increases with age, especially after 40."
)

# Convert user input into DataFrame with the correct column order
user_data = pd.DataFrame([user_input], columns=columns)

# Preprocess user input
scaled_input = scaler.transform(user_data)

# Function to generate report
def generate_report(user_input, prediction_proba, suggestions):
    report = FPDF()
    report.add_page()
    report.set_font("Arial", size=12)
    
    # Add title
    report.cell(200, 10, txt="Diabetes Prediction Report", ln=True, align='C')
    report.ln(10)
    
    # Add user inputs
    report.cell(200, 10, txt="User Inputs:", ln=True)
    for key, value in user_input.items():
        report.cell(200, 10, txt=f"{key}: {value}", ln=True)
    report.ln(10)
    
    # Add prediction results
    report.cell(200, 10, txt="Prediction Results:", ln=True)
    report.cell(200, 10, txt=f"Probability of diabetes: {float(prediction_proba[1]):.2f}", ln=True)
    report.cell(200, 10, txt=f"Probability of no diabetes: {float(prediction_proba[0]):.2f}", ln=True)
    report.ln(10)
    
    # Add suggestions
    report.cell(200, 10, txt="Personalized Suggestions:", ln=True)
    for suggestion in suggestions:
        report.cell(200, 10, txt=suggestion, ln=True)
    
    return report

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    # Display prediction result with color-coded risk level
    st.subheader("Prediction Result")
    risk_level = "High" if prediction_proba[1] > 0.7 else "Medium" if prediction_proba[1] > 0.4 else "Low"
    risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
    st.markdown(f"Risk Level: <span style='color:{risk_color};'>{risk_level}</span>", unsafe_allow_html=True)
    st.write(f"Probability of diabetes: {float(prediction_proba[1]):.2f}")
    st.write(f"Probability of no diabetes: {float(prediction_proba[0]):.2f}")

    # Progress bar for risk level
    st.progress(float(prediction_proba[1]))  # Convert float32 to float

    # LIME explainability with Plotly
    st.subheader("Explainability using LIME")
    try:
        explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(columns))),  # Placeholder data
            feature_names=columns,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification'
        )
        explanation = explainer.explain_instance(
            data_row=user_data.values[0],  # Use raw user input
            predict_fn=model.predict_proba
        )

        # Extract feature contributions from LIME explanation
        lime_contributions = []
        for feature, weight in explanation.as_list():
            # Extract the feature name by splitting on '>' or '<'
            feature_name = feature.split(' ')[0]
            lime_contributions.append((feature_name, weight))

        # Plot LIME explanation with Plotly
        lime_df = pd.DataFrame(lime_contributions, columns=["Feature", "Contribution"])
        fig = px.bar(lime_df, x="Contribution", y="Feature", orientation='h', 
                     title="LIME Feature Contributions", labels={"Contribution": "Contribution to Risk"})
        st.plotly_chart(fig)

        # Explain LIME results in simple terms
        st.write("Explanation of LIME Results:")
        st.write("The following features contributed the most to your diabetes risk prediction:")
        for feature, contribution in lime_contributions:
            st.write(f"- {feature}: Contribution to risk: {contribution:.2f}")
    except Exception as e:
        st.error(f"Error generating LIME explanation: {e}")

    # SHAP explainability with Plotly
    st.subheader("Explainability using SHAP")
    try:
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(user_data.values)  # Use raw user input

        # Use Plotly for SHAP summary plot
        fig = go.Figure()
        for i, col in enumerate(columns):
            fig.add_trace(go.Bar(x=[shap_values[0][i]], y=[col], orientation='h', name=col))
        fig.update_layout(title="SHAP Feature Contributions", xaxis_title="Contribution to Risk", yaxis_title="Feature")
        st.plotly_chart(fig)

        # Explain SHAP results in simple terms
        st.write("Explanation of SHAP Results:")
        st.write("The SHAP plot shows how each feature influenced the model's prediction:")
        for i, col in enumerate(columns):
            st.write(f"- {col}: Impact on prediction: {shap_values[0][i]:.2f}")
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {e}")

    # Personalized suggestions
    st.subheader("Personalized Suggestions")
    suggestions = []
    if user_input['Glucose'] > 140:
        suggestions.append("- Glucose: Your glucose level is high. Reduce sugar intake and monitor your diet.")
    if user_input['BMI'] > 25:
        suggestions.append("- BMI: Your BMI indicates overweight. Engage in regular exercise and maintain a healthy diet.")
    if user_input['BloodPressure'] > 80:
        suggestions.append("- Blood Pressure: Your blood pressure is elevated. Reduce salt intake and avoid stress.")
    if user_input['Age'] > 40:
        suggestions.append("- Age: As you age, the risk of diabetes increases. Regular health check-ups are recommended.")
    if user_input['Pregnancies'] > 5:
        suggestions.append("- Pregnancies: Multiple pregnancies may increase diabetes risk. Monitor glucose levels closely.")
    if user_input['SkinThickness'] > 40:
        suggestions.append("- Skin Thickness: Higher skin thickness may indicate higher body fat. Maintain a healthy diet and exercise routine.")
    if user_input['Insulin'] > 24.9:
        suggestions.append("- Insulin: Your insulin level is high. Avoid skipping meals and consult a healthcare provider.")
    if user_input['DiabetesPedigreeFunction'] > 1.0:
        suggestions.append("- Diabetes Pedigree Function: You have a strong family history of diabetes. Consider genetic counseling and regular screening.")

    if suggestions:
        st.write("Based on your inputs, here are some suggestions to manage your diabetes risk:")
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.write("Your inputs are within normal ranges. Keep maintaining a healthy lifestyle!")

    # Generate and download report
    report = generate_report(user_input, prediction_proba, suggestions)
    report_output = report.output(dest='S').encode('latin1')
    st.download_button(
        label="Download Report as PDF",
        data=report_output,
        file_name="diabetes_prediction_report.pdf",
        mime="application/pdf"
    )

    # Diabetic Diet Chart Section
    st.subheader("Diabetic Diet Chart")
    st.write("""
    Energy: 1400 Kcal | Protein: 50 gm

    7:00 am: Milk / Coffee / Tea (without sugar) - 1 cup  
    8:30 am: Any one of the mentioned Breakfast items:  
    - Phulka 1 nos / Idly 2 nos / Dhalia Upma / Jowar Upma - ½ cup  
    - Dosa / Pesarattu 1 no (with less oil & use non-stick pan) with Chutney / Sambar / Vegetables / Legumes curry - 1 cup  
    - Brown bread - 2 slices / Oats / Wheat flakes 1 tablespoon with milk + egg whites 1 / sprouts - ½ cup  

    11:00 am: Buttermilk / Soup / Vegetable Salad  
    1:00 pm: Vegetable salad  
    - Phulka / Jowar Roti (1 no) / Dhalia (wheat rava) - ½ cup  
    - Unpolished Rice - ½ cup (Raw wt gm)  
    - Dhal / Sambar - 1 cup  
    - Leafy vegetable - 1 cup  
    - Vegetable curry - 2 cups  
    - Curd - 1 cup  

    4:00 pm: Coffee / Tea / Milk - 1 cup / Nutrichoice / oats / multigrain biscuits - 2 nos / Fruit - 1 no  
    6:00 pm: Veg. soup / Buttermilk - 200ml  
    8:00 pm: Dinner same as lunch  
    9:30 pm: Buttermilk / Milk - 1 cup (without sugar)  
    """)

    # Foods to Avoid
    st.subheader("Foods to Avoid")
    st.write("""
    1. Sugar, jaggery, honey, jam, jellies, sweets, squashes, Proprietary drinks (Horlicks / Bournvita etc).  
    2. Bakery products (White bread, biscuits, cakes, doughnuts, buns, pizza, burger) & aerated beverages / cola drinks (Sprite, Pepsi etc). Maida and its products to be avoided.  
    3. Butter, Cheese, Ghee, Dalda / vanaspathi, margarine, Deep fried foods.  
    4. Egg yolk, red meats (cow, goat, sheep), organ meat (brain, liver, kidney, intestine), crabs, prawns, shrimps & lobsters.  
    5. Potato, Yam, Colacasia, Tapioca & sweet potato.  
    6. Banana, Sapota, Mango, Seethaphal, Jackfruit, green grapes, all fruit juices.  
    """)

    # Foods to Consume in Moderation
    st.subheader("Foods to Consume in Moderation")
    st.write("""
    1. Cooking oil not more than - 2 teaspoons/day.  
    2. Non-veg: Chicken (skinless) and fish 100gms twice a week.  
    """)

    # Allowed Fruits
    st.subheader("Allowed Fruits")
    st.write("""
    Any one of the below once a day (100 gms):  
    - Apple, Guava, Mosambi, Orange, Pear - 1 medium size  
    - Pomegranate - ½ fruit  
    - Watermelon - 2 slices or Musk melon - 1 cup  
    - Papaya - 1 long slice (7-8 cubes)  
    """)

    # Symptoms of Hypoglycemia
    st.subheader("Symptoms of Hypoglycemia")
    st.write("""
    - Weakness, sweating, excess hunger, headache, tremors  

    What to do?  
    - Take food/fruit with little added sugar immediately.  
    - In case symptoms don't subside, take sugar/sweet at once.  
    - Don't feed those who are unconscious.  
    """)