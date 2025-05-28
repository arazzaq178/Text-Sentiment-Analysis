# 🎬 IMDB Movie Reviews Sentiment Analysis

A machine learning project that builds a sentiment analysis model using the IMDB movie reviews dataset. The model classifies reviews as either **positive** or **negative** by applying natural language processing (NLP) and supervised learning techniques.

---

## 📂 Project Structure

```
.
├── IMDB Dataset.csv              # Raw dataset
├── sentiment_analysis.ipynb      # Main Jupyter Notebook
├── model.joblib                  # Saved trained model (optional)
├── tfidf_vectorizer.joblib       # Saved TF-IDF vectorizer (optional)
├── README.md                     # Project documentation
```

---

## 📌 Features

* Text Preprocessing (Tokenization, Stopwords removal, Lemmatization)
* Feature Extraction using TF-IDF
* Classification using Logistic Regression
* Evaluation with accuracy, precision, recall, and F1-score
* Model Saving and Loading using `joblib`
* Ready to deploy as a web app (GUI support possible)

---

## 🗃 Dataset

* **Source**: IMDB movie reviews
* **Columns**:

  * `review`: The full text of the movie review.
  * `sentiment`: Label indicating the sentiment (positive/negative).

---

## 🚀 How to Run

### 1. Clone the repository / Download files

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, here are the necessary packages:

```bash
pip install pandas numpy scikit-learn nltk joblib
```

### 3. Preprocess & Train

Open the `sentiment_analysis.ipynb` notebook and run all cells in sequence:

* Loads and cleans the dataset
* Preprocesses reviews
* Applies TF-IDF vectorization
* Trains a Logistic Regression model
* Evaluates performance

---

## 🧠 Model Evaluation

The model is evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**

Metrics are printed at the end of training for performance validation.

---

## 💾 Save & Load Model

Use `joblib` to persist the model and vectorizer:

```python
joblib.dump(model, 'model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
```

Later, load them via:

```python
model = joblib.load('model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')
```

---

## 🌐 Web App (Optional)

To turn this into a web app:

1. Use [Streamlit](https://streamlit.io/) or [Flask](https://flask.palletsprojects.com/).
2. Build an input form where users can type a review.
3. Load the model and TF-IDF vectorizer.
4. Display the predicted sentiment visually.

---

## 🧠 Skills Gained

* Natural Language Processing (NLP)
* Feature Engineering with TF-IDF
* Model Evaluation Techniques
* Serialization with `joblib`
* Data Cleaning & Preprocessing in Python

---

## 🛠 Tools & Libraries

* Python
* Pandas, NumPy
* NLTK
* Scikit-learn
* Joblib

---

## 📌 Author

**Abdul Razzaq**
Associate Degree in Data Science
Virtual University
GitHub: \arazzaq178
