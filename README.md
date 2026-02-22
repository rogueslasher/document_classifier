# Document Classification with Active Learning

An NLP system that classifies documents into 20 categories using Active Learning, reducing labeling effort by 35% while achieving 85.3% accuracy.

This project demonstrates how uncertainty-based sample selection improves model performance while requiring fewer labeled examples.

Built using Python, scikit-learn, NLTK, and Streamlit.

---

## Overview

Text classification models typically require large labeled datasets, which are expensive and time-consuming to create.

This project implements Active Learning using uncertainty sampling, where the model iteratively selects the most informative samples to label, improving learning efficiency and performance.

Key outcomes:

- Final accuracy: 85.3%
- +3.2% accuracy improvement over random sampling
- 35% reduction in labeled data required
- Multi-class classification across 20 categories

Dataset used: 20 Newsgroups (11,314 training documents)

---

## Features

- Active Learning implementation using uncertainty sampling
- Complete NLP pipeline from preprocessing to evaluation
- TF-IDF feature extraction
- Logistic Regression classifier
- Model evaluation and performance analysis
- Visualization of learning performance
- Streamlit interface for interactive predictions

---

## Tech Stack

- Python
- scikit-learn
- NLTK
- Streamlit
- matplotlib
- seaborn
- NumPy
- pandas

---

## Results

| Metric | Active Learning | Random Sampling |
|--------|----------------|----------------|
| Accuracy | 85.3% | 82.1% |
| F1 Score | 0.847 | 0.814 |
| Label Efficiency | 35% fewer labels | — |

Active Learning achieves better performance with significantly fewer labeled examples.

---

## Architecture

Pipeline workflow:

```
Raw Text
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Baseline Model Training
   ↓
Active Learning Loop
   ↓
Model Evaluation and Analysis
```

Uncertainty sampling formula:

```
uncertainty = 1 − max(predicted_probability)
```

Samples with highest uncertainty are selected for labeling.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rogueslasher/document-classifier-active-learning.git
cd document-classifier-active-learning
```

Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK resources:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

---

## Running the Project

Open the notebooks in order:

```
notebooks/
│
├── explore_data.ipynb
├── text_preprocess.ipynb
├── feature_engineering.ipynb
├── baseline_model.ipynb
├── active_learning.ipynb
└── analysis.ipynb
```

Run each notebook sequentially to execute the full pipeline.

To run the Streamlit interface:

```bash
streamlit run app.py
```

---

## Project Structure

```
document-classifier-active-learning/
│
├── notebooks/
│   ├── explore_data.ipynb
│   ├── text_preprocess.ipynb
│   ├── feature_engineering.ipynb
│   ├── baseline_model.ipynb
│   ├── active_learning.ipynb
│   └── analysis.ipynb
│
├── notebooks/
│   ├── complete_analysis.png
│   ├── per_class_performance.png
│   ├── project_summary.txt
│   └── sample_data.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Key Learnings

- Active Learning implementation in real-world classification
- Natural Language Processing pipeline development
- TF-IDF feature engineering
- Model evaluation using accuracy and F1 score
- Efficient data utilization strategies
- End-to-end machine learning workflow

---

## Future Improvements

- Deploy using Streamlit Cloud or AWS
- Implement BERT-based classification
- Add FastAPI backend for inference
- Implement model explainability using SHAP or LIME
- Convert notebooks into production pipeline scripts

---

## Author

Aniket Pandey  
GitHub: https://github.com/rogueslasher