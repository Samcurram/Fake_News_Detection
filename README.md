# Fake_News_Detection
# 🕵️ Truth or Trash: Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
In an era of misinformation, identifying fake news is critical. "Truth or Trash" is a Machine Learning project that utilizes Natural Language Processing (NLP) to classify news articles as **Real** or **Fake**. 

The system is trained on the ISOT Fake News Dataset using **TF-IDF Vectorization** and **Logistic Regression**, achieving high accuracy in distinguishing between legitimate news sources and fabricated stories.

## 🚀 Key Features
* **Text Preprocessing:** Automated cleaning pipeline (Stemming, Stop-word removal, Regex cleaning).
* **TF-IDF Vectorization:** Converts textual data into numerical statistics.
* **Binary Classification:** Uses Logistic Regression to predict truthfulness.
* **Performance Metrics:** Evaluation using Confusion Matrix and Accuracy Scores.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, Seaborn, Matplotlib
* **Environment:** Google Colab / Jupyter Notebook

## 📊 Dataset
The model was trained on the **ISOT Fake News Dataset** (or similar Kaggle dataset), containing:
* **True.csv:** Articles from legitimate sources (e.g., Reuters).
* **Fake.csv:** Fabricated articles.
* *Note: Dataset is not included in this repo due to size constraints. Download it [here](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).*

## 📈 Results
The model achieved an accuracy of **~99%** on the test set.

**Confusion Matrix:**
*(Upload your screenshot here, e.g., ![Confusion Matrix](![Uploading Screenshot 2025-12-09 210745.png…]()
))*

## 💻 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/Truth-Or-Trash-Fake-News-Detection.git](https://github.com/your-username/Truth-Or-Trash-Fake-News-Detection.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `Fake_News_Detection.ipynb` in Jupyter or Google Colab and run all cells.

## 🔮 Future Improvements
* Implement LSTM (Deep Learning) for better context understanding.
* Create a web interface using Streamlit or Flask.
* Deploy the model as a real-time API.

## 👤 Author
**[Abdul Salaam]**
* [www.linkedin.com/in/abdus-salaam]
* [https://github.com/Samcurram]
