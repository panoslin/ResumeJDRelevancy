import os
import re
import time

import pandas as pd
import pdfplumber
import torch
import torch.nn as nn
from datasets import (
    Dataset,
    DatasetDict,
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader


def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    with pdfplumber.open(file_path) as pdf:
        text = ''.join(page.extract_text() for page in pdf.pages)
    return text


def load_and_preprocess_data(job_desc_csv_path, resume_pdf_path):
    """Loads and preprocesses data."""
    df = pd.read_csv(job_desc_csv_path)
    df['resume'] = extract_text_from_pdf(resume_pdf_path)
    df['label'] = df['label'].astype(int)
    df.dropna(subset=['resume', 'job_description', 'label'], inplace=True)

    # Apply preprocessing
    df['resume'] = df['resume'].apply(preprocess_text)
    df['job_description'] = df['job_description'].apply(preprocess_text)
    return df


def preprocess_text(text):
    """Preprocesses the input texts if necessary."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def create_dataset_splits(df):
    """Creates dataset splits."""
    dataset = Dataset.from_pandas(df)
    # Split into train and temp (which will be split into validation and test)
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    # Split the temp set into validation and test
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # Combine splits into a DatasetDict
    dataset_splits = DatasetDict({
        'train':      train_testvalid['train'],
        'validation': test_valid['train'],
        'test':       test_valid['test']
    })
    return dataset_splits


def get_device():
    """Returns the appropriate device (CPU, CUDA, or MPS)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


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


def build_model(device, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=2):
    """Builds and returns the model."""
    base_model = SentenceTransformer(model_name)
    base_model.to(device)
    model = SBERTClassifier(base_model, num_classes)
    model.to(device)
    return model


class ResumeJobDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        resume = item['resume']
        job_description = item['job_description']
        label = item['label']
        return (resume, job_description), label


def get_dataloaders(dataset_splits, batch_size=16):
    """Returns train, validation, and test dataloaders."""
    train_dataset = ResumeJobDataset(dataset_splits['train'])
    validation_dataset = ResumeJobDataset(dataset_splits['validation'])
    test_dataset = ResumeJobDataset(dataset_splits['test'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, validation_dataloader, test_dataloader


def train(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_pairs, labels = batch
        labels = labels.to(device)
        # Zero Gradients
        optimizer.zero_grad()
        # Forward Pass
        logits = model(input_pairs)
        # Compute Loss
        loss = criterion(logits, labels)
        # Backward Pass
        loss.backward()
        # Update Parameters
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluates the model."""
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_pairs, labels = batch
            labels = labels.to(device)
            # Forward Pass
            logits = model(input_pairs)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # Compute Metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, predictions)
    return accuracy, f1, recall, mcc


def main(model_name='models/best_model.pt', evaluate_only=False):
    # Load and preprocess data
    df = load_and_preprocess_data('dataset/job_descriptions.csv', 'dataset/resume.pdf')
    dataset_splits = create_dataset_splits(df)
    print(f"Train size: {len(dataset_splits['train'])}")
    print(f"Validation size: {len(dataset_splits['validation'])}")
    print(f"Test size: {len(dataset_splits['test'])}")

    # Setup device and model
    device = get_device()
    model = build_model(device)

    # Prepare data loaders
    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(dataset_splits)

    if not evaluate_only:
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

        num_epochs = 10
        best_validation_recall = 0
        start_epoch = 0

        # Check for existing checkpoint
        checkpoint_path = 'models/checkpoint.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_validation_recall = checkpoint['best_validation_recall']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Starting training from scratch")

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = train(model, train_dataloader, optimizer, criterion, device)
            val_accuracy, val_f1, val_recall, val_mcc = evaluate(model, validation_dataloader, device)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}, "
                  f"Validation Recall: {val_recall:.4f}, Validation MCC: {val_mcc:.4f}")

            # Save the model if it has the best recall rate so far
            if val_recall > best_validation_recall:
                best_validation_recall = val_recall
                torch.save(model.state_dict(), model_name)
                print("Saved new best model with Validation Recall: {:.4f}".format(best_validation_recall))

            # Save checkpoint after each epoch
            checkpoint = {
                'epoch':                  epoch,
                'model_state_dict':       model.state_dict(),
                'optimizer_state_dict':   optimizer.state_dict(),
                'best_validation_recall': best_validation_recall
            }
            torch.save(checkpoint, checkpoint_path)

    # Load the best model
    model.load_state_dict(torch.load(model_name, map_location=device, weights_only=False))

    # Evaluate on test set
    test_accuracy, test_f1, test_recall, test_mcc = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, "
          f"Test Recall: {test_recall:.4f}, Test MCC: {test_mcc:.4f}")


if __name__ == "__main__":
    start_time = time.time()
    main(model_name='models/best_model_2024_11_10_07.pt', evaluate_only=False)
    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
