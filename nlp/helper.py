import re

import pdfplumber
import torch
from sentence_transformers import SentenceTransformer

from nlp.SBERTClassifier import SBERTClassifier


def get_device():
    """Returns the appropriate device (CPU, CUDA)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def preprocess_text(text):
    """Preprocesses the input texts if necessary."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text


def build_model(device, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=2):
    """Builds and returns the model."""
    base_model = SentenceTransformer(model_name)
    base_model.to(device)
    model = SBERTClassifier(base_model, num_classes)
    model.to(device)
    return model
