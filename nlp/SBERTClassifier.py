import torch
import torch.nn as nn


class SBERTClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(SBERTClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.get_sentence_embedding_dimension(), num_classes)

    def forward(self, input_pairs):
        resumes, job_descriptions = input_pairs
        # Tokenize the inputs
        encoded_inputs = self.base_model.tokenize(resumes + job_descriptions)
        # Move inputs to the device
        device = next(self.base_model.parameters()).device
        encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
        # Get embeddings
        model_output = self.base_model(encoded_inputs)
        embeddings = model_output['sentence_embedding']
        # Split embeddings back into resumes and job descriptions
        half = len(embeddings) // 2
        resume_embeddings = embeddings[:half]
        job_embeddings = embeddings[half:]
        # Compute the absolute difference between embeddings
        features = torch.abs(resume_embeddings - job_embeddings)
        # Pass through classifier
        logits = self.classifier(features)
        return logits
