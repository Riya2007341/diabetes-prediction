import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "The Patient has Diabetes"
    else:
        result = "The Patient has no Diabetes"


    # Dataset distribution chart
    counts = df["Outcome"].value_counts()

    labels = ["Non-Diabetic", "Diabetic"]
    sizes = [
        counts.get(0, 0),
        counts.get(1, 0)
    ]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.08),
        shadow=True
    )

    ax.set_title("Diabetes Dataset Distribution")


    return result, fig



# Gradio Interface
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
        gr.Number(label="Age"),
    ],

    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Plot(label="Dataset Distribution")
    ],

    title="🩺 Diabetes Prediction System",

    description="""
Predict whether a patient has diabetes using a Machine Learning (KNN) model.

Enter the patient's medical information and click Submit.
"""
)


diabetes_app.launch(
    server_name="0.0.0.0",
    server_port=7880
)
