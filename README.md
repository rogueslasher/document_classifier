# Document Classification with Active Learning

An NLP system that classifies documents into 20 categories using Active Learning, reducing labeling effort by 35% while achieving 85%+ accuracy.

Built using Python, scikit-learn, NLTK, and Streamlit.

---

## Overview

Traditional text classification requires large labeled datasets, which are expensive and time-consuming.

This project implements Active Learning with uncertainty sampling, allowing the model to select the most informative samples and learn more efficiently.

**Key outcomes:**

- 35% fewer labeled examples required  
- +3.2% accuracy improvement vs random sampling  
- Final accuracy: 85.3%  
- 20-class document classification  

Dataset: 20 Newsgroups (11,314 training documents)

---

## Features

- Active Learning using uncertainty sampling  
- End-to-end NLP pipeline (preprocessing → feature engineering → training)  
- TF-IDF vectorization  
- Logistic Regression classifier  
- Model saving and loading  
- Interactive Streamlit interface for predictions  

---

## Tech Stack

- Python  
- scikit-learn  
- NLTK  
- Streamlit  
- matplotlib  
- seaborn  
- TF-IDF  
- Logistic Regression  

---

## Results

| Metric | Active Learning | Random Sampling |
|--------|----------------|----------------|
| Accuracy | 85.3% | 82.1% |
| F1 Score | 0.847 | 0.814 |
| Label Efficiency | 35% fewer labels | — |

---

## Architecture

Pipeline:

```
Text Input
   ↓
Preprocessing (tokenization, lemmatization)
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Active Learning Loop (uncertainty sampling)
   ↓
Final trained model
```

Uncertainty formula:

```
uncertainty = 1 − max(predicted_probability)
```

---

## Installation

```bash
git clone https://github.com/rogueslasher/document-classifier-active-learning.git
cd document-classifier-active-learning

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Download NLTK data:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## Run Training Pipeline

```bash
python step3_text_preprocessing.py
python step4_feature_engineering.py
python step5_baseline_model.py
python step6_active_learning.py
```

Run demo app:

```bash
streamlit run app.py
```

---

## Live Demo

Deployment coming soon.

---

## Project Structure

```
document-classifier-active-learning/
│
├── step3_text_preprocessing.py
├── step4_feature_engineering.py
├── step5_baseline_model.py
├── step6_active_learning.py
├── app.py
├── requirements.txt
├── final_model.pkl
└── README.md
```

---

## Key Learnings

- Active Learning implementation  
- NLP preprocessing and TF-IDF  
- Model evaluation and optimization  
- Building deployable ML pipelines  
- Streamlit integration  

---

## Future Work

- Deploy using Streamlit Cloud or AWS  
- Add BERT-based classifier  
- Build FastAPI backend  
- Add explainability using SHAP or LIME  

---

## Author

Aniket Pandey  
GitHub: https://github.com/rogueslasher