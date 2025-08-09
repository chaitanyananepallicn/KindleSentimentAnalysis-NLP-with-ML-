# Kindle Review Sentiment Analyzer

A collection of web applications built with Python and Streamlit to classify the sentiment of Kindle book reviews as either "Positive" or "Negative". This project explores and deploys three different Natural Language Processing models: Bag-of-Words, TF-IDF, and Average Word2Vec.

***

## Live Applications

**You can access live deployed demos for each model here:**

* [**➡️ Live Demo: Bag-of-Words Model**](https://your-bow-app-url-here.com)
* [**➡️ Live Demo: TF-IDF Model**](https://your-tfidf-app-url-here.com)
* [**➡️ Live Demo: Word2Vec Model**](https://your-word2vec-app-url-here.com)

***

## 🚀 Features

* **Three Independent Models**: Test and compare the performance of BoW, TF-IDF, and Word2Vec on the same task.
* **Interactive UI**: A clean and simple web interface for each model, built with Streamlit.
* **Real-time Sentiment Prediction**: Instantly classifies any user-provided Kindle review.
* **Confidence Score**: Displays each model's confidence in its prediction with a progress bar.
* **Efficient Backend**: Uses pre-trained models saved as pickle files for fast loading and prediction.

***

## 🛠️ Tech Stack

* **Core Language**: Python
* **Web Framework**: Streamlit
* **ML & NLP Libraries**:
    * `Scikit-learn` for machine learning (CountVectorizer, TfidfVectorizer, Classifiers).
    * `NLTK` for natural language processing (stopwords, lemmatization).
    * `Gensim` for Word2Vec model training and loading.
    * `BeautifulSoup4` for HTML parsing during text preprocessing.

***

## ⚙️ Getting Started

To run this project on your local machine, follow these steps.

### Prerequisites

Make sure you have Python 3.8+ and pip installed on your system.

### Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Create and Activate a Virtual Environment**
    * **On macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Create a `requirements.txt` file**
    * Create a file named `requirements.txt` and add the following lines:
        ```text
        streamlit
        scikit-learn
        nltk
        gensim
        beautifulsoup4
        ```

4.  **Install the Required Libraries**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Ensure All Model Files are Present**
    * Make sure all your trained model files are in the root directory of the project, as shown in the Project Structure below.

6.  **Handle Large Model Files (Git LFS)**
    * Model files, especially Word2Vec models, can be larger than the 25 MB limit allowed by services like GitHub. To handle this, you need to use Git Large File Storage (LFS).

    * **Install Git LFS:** First, [download and install the Git LFS extension](https://git-lfs.github.com/).

    * **Set up LFS in your repository:** Run the following commands from the root of your project folder.
        ```bash
        # Initialize Git LFS (only needs to be done once per repository)
        git lfs install

        # Tell LFS which files to track. It's best to track by file extension.
        git lfs track "*.model"
        git lfs track "*.pickle"
        git lfs track "*.npy"

        # Make sure the .gitattributes file is tracked by Git
        git add .gitattributes
        ```
    * Now you can commit and push your large files as you normally would. Git LFS will handle them automatically.
        ```bash
        git add .
        git commit -m "Add models and configure LFS"
        git push origin main
        ```

7.  **Run a Streamlit Application**
    * You can run any of the three apps. Use one of the following commands in your terminal:

    * To run the Bag-of-Words app:
        ```bash
        streamlit run app_bow.py
        ```

    * To run the TF-IDF app:
        ```bash
        streamlit run app_tfidf.py
        ```

    * To run the Word2Vec app:
        ```bash
        streamlit run app_word2vec.py
        ```
    * After running a command, open your web browser and navigate to the local URL provided (usually `http://localhost:8501`).

***

## 📂 Project Structure

This shows how all files for the three models can be organized in a single project folder.

```text
.
├── app_bow.py              # Streamlit app for Bag-of-Words model
├── app_tfidf.py            # Streamlit app for TF-IDF model
├── app_word2vec.py         # Streamlit app for Word2Vec model
│
├── BoW.pickle              # Saved CountVectorizer object
├── TfIdf.pickle            # Saved TfidfVectorizer object
│
├── model.pickle            # Saved classifier for BoW/TF-IDF
├── model_w2v.pickle        # Saved classifier for Word2Vec
│
├── word2vec.model          # Saved Gensim Word2Vec model files
├── word2vec.model.syn1neg.npy
└── word2vec.model.wv.vectors.npy
│
├── .gitattributes          # Git LFS tracking configuration file
├── requirements.txt        # Python dependencies
└── README.md               # This file
