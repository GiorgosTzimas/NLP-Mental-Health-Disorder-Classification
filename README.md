NLP for Mental Health Disorder Classification

This project applies NLP techniques to identify and classify **mental health conditions** based on user-generated text data. It explores traditional machine learning, deep learning, and transformer-based approaches to predict the likelihood of disorders including **anxiety, depression, bipolar disorder**, and others.

## Dataset

The dataset that utilized can be found here:
<a href="https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data" style="text-decoration: underline;">here</a>

**Description**: The dataset contains labeled text posts from individuals experiencing various mental health issues. Each entry includes:

- `text`: User-submitted post
- `label`: Mental health condition (e.g., depression, anxiety, normal etc.)

## Objective

To build models that accurately **classify mental health conditions** using NLP techniques and evaluate their performance across different approaches:

- Traditional Machine Learning
- Deep Learning (Neural Networks)
- Transformer Models (BERT)
- Large Language Models (LLMs)

---

## Challenges 

- Overfitting
- Class imbalance

---

## Preprocessing

- Text normalization (lowercasing, punctuation removal)
- Tokenization
- Stopword removal
- Lemmatization
- Vectorization:
  - **BoW**
  - **TF-IDF**
  - **Trigrams**
  - **Word embeddings** (Word2Vec/BERT embeddings)

---

## Models Implemented

### Traditional ML Models
- Logistic Regression
- Naive Bayes
- Random Forest

### Deep Learning
- Multilayer Perceptron (MLP)

### Transformer Models
- **BERT** (Fine-tuned using Hugging Face Transformers)
- Evaluation with attention to multi-class classification

### LLM (GPT-4o)
- Prompt-based classification or zero-shot/few-shot learning using GPT-style models

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy
- Precision, Recall, F1-score (per class and macro/micro averages)
- Confusion Matrix

---

## Results Summary

| Model                 | Accuracy | Weighted F1 | Notes                      |
|-----------------------|----------|-------------|----------------------------|
| LR (Trigram & TF-IDF) |     0.72 |        0.73 | Strong baseline            |
| MLP - Bert            |     0.78 |        0.78 | Best performance           |
| LLM (GPT-4o)          |          |        0.69 | Great few-shot performance |

> **Insight**: Fine-tuned BERT provided the best trade-off between accuracy and interpretability for multi-class classification.
