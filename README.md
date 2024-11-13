## Task 
Predict if a resume matches a job description using binary classification.
## Approach
Use a pre-trained SBERT model to obtain embeddings and build a classifier on top.
## Key Steps:
1. Prepare data and split into training, validation, and test sets.
2. Define a custom model that integrates SBERT embeddings and a classification layer.
3. Handle device assignments to avoid computation errors.
4. Train the model using an appropriate loss function and optimizer.
5. Evaluate the model’s performance and adjust as necessary.

## Current training results
- Train with base model [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
  - Train size: 512
  - Validation size: 64
  - Test size: 65
- Metrics:
  - Test Accuracy: 0.8615 
  - Test F1 Score: 0.8696
  - Test Recall: 0.9091
  - Test MCC: 0.7257
- Total time: 7.692301034927368 seconds

## TODO:
- Collect more data