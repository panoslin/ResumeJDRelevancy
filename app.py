from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_evaluator import (
    predict,
    load_trained_model,
)
from model_trainer import (
    get_device,
    preprocess_text,
    extract_text_from_pdf,
)


class PredictionRequest(BaseModel):
    resume_text: str = None
    resume_pdf_path: str = None
    job_description_text: str


class PredictionResponse(BaseModel):
    predicted_class: int
    confidence: float
    message: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.linkedin.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = None
base_model_name = "BAAI/bge-small-en-v1.5"
model_path = "models/best_model_2024_11_11_00_42.pt"


@app.on_event("startup")
def load_model():
    global model, device
    device = get_device()
    model = load_trained_model(model_path, device, base_model_name)
    model.to(device)
    model.eval()
    print("Model loaded successfully")


@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    try:
        if request.resume_text:
            resume_text = preprocess_text(request.resume_text)
        elif request.resume_pdf_path:
            resume_text = extract_text_from_pdf(request.resume_pdf_path)
            resume_text = preprocess_text(resume_text)
        else:
            return PredictionResponse(
                predicted_class=-1,
                confidence=0.0,
                message="No resume provided."
            )

        job_description_text = preprocess_text(request.job_description_text)

        predicted_class, confidence = predict(
            resume_text,
            job_description_text,
            model_path,
            device,
            base_model_name
        )

        if predicted_class == 1:
            message = f"The resume is compatible with the job description (Confidence: {confidence:.2f})"
        else:
            message = f"The resume is not compatible with the job description (Confidence: {confidence:.2f})"

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            message=message
        )

    except Exception as e:
        return PredictionResponse(
            predicted_class=-1,
            confidence=0.0,
            message=f"An error occurred during prediction: {str(e)}"
        )
