import gradio as gr
import joblib

model = joblib.load("model.pkl")

def predict(g1, g2, studytime, absences, failures):

    prediction = model.predict(
        [[g1, g2, studytime, absences, failures]]
    )

    return round(prediction[0], 2)

app = gr.Interface(
    fn=predict,

    inputs=[
        gr.Slider(0, 20, label="G1"),
        gr.Slider(0, 20, label="G2"),
        gr.Slider(1, 4, label="Study Time"),
        gr.Slider(0, 100, label="Absences"),
        gr.Slider(0, 4, label="Failures")
    ],

    outputs=gr.Textbox(
        label="Predicted Final Grade (G3)"
    ),

    title="Student Performance Prediction",

    description="Predict student final grade using Linear Regression."
)

app.launch()