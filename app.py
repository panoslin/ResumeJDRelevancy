import json
import uuid
from typing import Optional

import boto3
import torch.nn as nn
from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nlp.evaluator import (
    predict,
    load_trained_model,
)
from nlp.helper import (
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client('s3')
BUCKET_NAME = "resume-jd-relevancy-data"

model: Optional[nn.Module] = None
device = None
base_model_name = "BAAI/bge-small-en-v1.5"
model_path = "models/best_model.pt"


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
            device,
            model
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


def upload_json_to_s3(file_content, file_name):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=f"raw/{file_name}",
        Body=file_content,
        ContentType='application/json'
    )


class RawTrainData(BaseModel):
    resume_text: str
    job_description: str
    applied: bool
    # website complete link
    link: str
    # website domain
    source: str
    # Google user id
    user: Optional[str] = None
    # job id | link
    job_id: Optional[str] = None


@app.post("/applications", response_model=dict, status_code=201)
async def upload_resume(payload: RawTrainData):
    try:
        random_file_name = f"{uuid.uuid4().hex}.json"
        data = payload.dict()
        json_data = json.dumps(data)
        upload_json_to_s3(json_data, random_file_name)
        return {
            "message":   "File uploaded successfully",
            "file_name": random_file_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
