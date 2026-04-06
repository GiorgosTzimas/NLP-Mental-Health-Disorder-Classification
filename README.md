# NLP for Mental Health Disorder Classification

This project applies NLP techniques to identify and classify **mental health conditions** based on user-generated text data. It explores traditional machine learning, deep learning, and transformer-based approaches to predict the likelihood of disorders including **anxiety, depression, bipolar disorder**, and others.

The project was developed as part of academic coursework for the course Natural Language Processing and Text Analytics (MSc. in Business Administration and Data Science).


## Dataset

Source: Kaggle
The dataset that utilized can be found
<a href="https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data" style="text-decoration: underline;">here</a>

**Description:**
- ~50,000 text samples  
- User-generated social media posts labeled by mental health condition  

**Features:**
- `text` → user-submitted content  
- `label` → condition (e.g., depression, anxiety, normal)


## Approach

The project follows a standard NLP pipeline:

**Preprocessing:**
- Text cleaning (URLs, mentions, punctuation removal)  
- Lowercasing  
- Tokenization and lemmatization  
- Train-test split (80–20) to avoid data leakage

**Feature representations:**
- Bag of Words (BoW)  
- TF-IDF  
- Trigrams  
- Word2Vec embeddings  
- BERT embeddings

**Modeling:**
- Traditional ML models used as baseline  
- Neural networks for dense embeddings  
- Fine-tuned BERT for contextual learning  
- GPT-4o for zero-shot and few-shot classification
  

## Models Implemented

**Traditional ML:**
- Logistic Regression  
- Naive Bayes  
- Random Forest  

**Deep Learning:**
- Multilayer Perceptron (MLP)  

**Transformer Models:**
- BERT (fine-tuned using Hugging Face)  

**LLM:**
- GPT-4o (zero-shot vs few-shot prompting)

## Evaluation

Models were evaluated using:

- Accuracy  
- Precision, Recall, F1-score  
- Macro and weighted averages  
- Confusion Matrix  

Special attention was given to **F1-score** due to class imbalance.

## Results Summary

| Model                 | Accuracy | Weighted F1 | Notes                      |
|-----------------------|----------|-------------|----------------------------|
| LR (Trigram & TF-IDF) | 0.72     | 0.73        | Strong baseline            |
| MLP - BERT            | 0.78     | 0.78        | Best performance           |
| LLM (GPT-4o)          | –        | 0.69        | Strong few-shot results    |

> **Insight:** Fine-tuned BERT achieved the best overall performance, especially for minority classes.


## Key Insights

- Logistic Regression is a strong baseline with optimized vectorization  
- Trigram features improve performance over unigrams  
- Dense embeddings require more expressive models to be effective  
- Fine-tuned BERT significantly improves classification, especially for underrepresented classes  
- LLMs perform well with minimal setup but come with higher cost and lower consistency  


## Practical Implications

- NLP models can support early detection of mental health signals from text  
- Transformer models are better suited for complex, context-dependent tasks  
- LLMs provide a flexible alternative when training is not feasible  
- Model choice should balance performance, cost, and deployment constraints  


## Tools Used

- Python  
- scikit-learn  
- TensorFlow / PyTorch  
- Hugging Face Transformers  
- OpenAI API (GPT-4o)  


## Project Structure

```
nlp-mental-health-classification/
│
├── README.md
├── mental_health_nlp.ipynb
└── mental_health_nlp_report.pdf
```


## Report

A detailed explanation of preprocessing, modeling, and evaluation is available in: `mental_health_nlp_report.pdf`
