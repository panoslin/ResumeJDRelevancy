import torch

from model_trainer import (
    build_model,
    get_device,
    preprocess_text,
    extract_text_from_pdf,
)


def load_trained_model(model_path, device):
    """Loads the trained model from the specified path."""
    # Build the model architecture
    model = build_model(device)
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()  # Set the model to evaluation mode
    return model


def predict(resume_text, job_description_text, model_path, device):
    """Predicts the compatibility between a resume and a job description."""
    # Preprocess the texts
    resume_text, job_description_text = preprocess_text(resume_text), preprocess_text(job_description_text)
    model = load_trained_model(model_path, device)
    # Move model to device
    model.to(device)

    # Prepare the inputs
    input_pairs = ([resume_text], [job_description_text])  # Wrap texts in lists

    # Tokenize the inputs using the model's base tokenizer
    encoded_inputs = model.base_model.tokenize([resume_text, job_description_text])

    # Move inputs to device
    encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}

    # Get embeddings
    with torch.no_grad():
        model_output = model.base_model(encoded_inputs)
        embeddings = model_output['sentence_embedding']

    # Split embeddings back into resume and job description
    resume_embedding = embeddings[0].unsqueeze(0)  # Shape: [1, embedding_dim]
    job_embedding = embeddings[1].unsqueeze(0)  # Shape: [1, embedding_dim]

    # Compute the absolute difference between embeddings
    features = torch.abs(resume_embedding - job_embedding)

    # Pass through classifier
    logits = model.classifier(features)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

    return predicted_class, confidence


def main(job_description_text, model_path='models/best_model.pt', resume_pdf_path='dataset/resume.pdf'):
    device = get_device()
    resume_text = extract_text_from_pdf(resume_pdf_path)

    # Predict compatibility
    predicted_class, confidence = predict(resume_text, job_description_text, model_path, device)

    # Interpret the prediction
    if predicted_class == 1:
        print(f"The resume is compatible with the job description (Confidence: {confidence:.2f})")
    else:
        print(f"The resume is not compatible with the job description (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    with open("model_eveluate_job_description.txt", "r") as f:
        evaluate_jd = f.read()

    main(
        job_description_text=evaluate_jd,
        model_path="models/best_model_2024_11_10_07.pt",
        resume_pdf_path="dataset/resume.pdf",
    )
