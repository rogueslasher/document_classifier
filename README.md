# 📄 Document Classifier with Active Learning & Explainable AI (LIME)

A premium, interactive Machine Learning web application that demonstrates **Active Learning** (Human-in-the-Loop) and **Explainable AI (XAI)** on the 20 Newsgroups text classification dataset. Built with Python, Streamlit, Scikit-Learn, NLTK, and Altair.

---

## 🌟 Key Features

### 1. 🔮 Explainable Document Classification
*   **Real-time Inference:** Paste any text, and the baseline Logistic Regression model classifies it into one of 20 categories.
*   **Local Explainability Highlights:** Uses TF-IDF feature weights and model coefficients to calculate word contributions. Influential words are highlighted in color:
    *   **Vibrant Green:** Words that drove the classification *towards* the predicted class (positive influence).
    *   **Vibrant Red:** Words that drove the classification *away* from the predicted class (negative influence).
*   **Interactive Tooltips:** Hover over any highlighted word to inspect its numeric contribution score.

### 2. ⚙️ Interactive Active Learning Studio (Human-in-the-Loop)
*   **Unlabeled Pool:** Loads the raw unlabeled training corpus and holds it in the background.
*   **Uncertainty Sampling Strategies:** Select how to sample documents for labeling:
    *   **Entropy:** Selects documents with the highest prediction uncertainty across all 20 classes.
    *   **Margin:** Selects documents with the smallest confidence margin between the top two predicted classes.
    *   **Random:** Selects a random document.
*   **Live Retraining:** Submit the correct category label for the document. The model instantly retrains on the updated training set, evaluates itself against the test set, and updates its accuracy curve and stats dynamically.

### 3. 📈 Advanced Interactive Analytics (Altair)
*   **Interactive Learning Curves:** Compare the training efficiency of Active Learning vs. Random Sampling. Your live labeling session is plotted dynamically on the chart.
*   **Confusion Matrix Heatmap:** An interactive 20x20 confusion matrix. Hover over any cell to see the `Actual Category`, `Predicted Category`, and `Document Count` to identify which classes are most commonly confused by the model.

### 4. 📂 Bulk Document Classification
*   **CSV File Upload:** Process batches of documents by uploading a CSV.
*   **Column Selection:** Select which column in the CSV contains the raw document texts.
*   **Interactive Previews & Export:** Watch predictions finish with a live progress bar, inspect a 10-row preview, analyze the predicted category distribution chart, and download the results as a new CSV.

### 5. 🎨 Modern Design Aesthetics
*   **Custom Typography:** Integrates Google's *Outfit* font for polished typography.
*   **Styled Cards:** Metrics cards feature modern shadow styling, borders, and smooth upward-float hover transitions.
*   **Gradients:** Headers and highlights use sleek modern gradients.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rogueslasher/document_classifier.git
    cd document_classifier
    ```

2.  **Activate the Virtual Environment:**
    *   On Windows:
        ```powershell
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirement.txt
    ```

4.  **Run the Web App:**
    ```bash
    streamlit run app.py
    ```

---

## 🔬 Model Details
*   **Features:** TF-IDF Vectorizer (Max features: 5,000, unigrams, minimum document frequency = 5).
*   **Classifier:** Logistic Regression (L-BFGS solver, multi-class, L2 regularization).
*   **Dataset:** 20 Newsgroups (10,994 total documents split 80/20 into train/test sets).
