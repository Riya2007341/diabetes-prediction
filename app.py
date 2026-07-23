import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_mode.pkl")


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
):
    
    # Create input array
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    # Predict
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "The patient has a high risk of developing diabetes."
    else:
        return "The patient has a low risk of developing diabetes."


# Create Gradio interface
diabetes_app = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Risk Prediction",
    description="Predict whether a patient has a high or low risk of developing diabetes.",
)

# Launch app
diabetes_app.launch(
    server_name="0.0.0.0",
    server_port=8080,
)
      
