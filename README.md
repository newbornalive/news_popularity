# news_popularity
This project combines the strengths of two previous studies to build a comprehensive machine learning pipeline that predicts the popularity of online news articles based on metadata, content features, and publication details. We tackle this using both regression and classification approaches, with advanced feature selection, model tuning, and natural language processing (NLP) methods.

Problem Statement

Predict the popularity of news articles, measured by number of social media shares, using:

    Structural features (e.g., article length, channel, images)
    Publication metadata (e.g., weekday, weekend)
    Text-based content features (e.g., keywords, NLP bag-of-words)

We use:

    Regression: Predict number of shares
    Classification: Predict if an article is “popular” based on a threshold (e.g., top 50%, 90%, 99%)

Dataset Description

    Source: UCI Online News Popularity Dataset
    Total Records: ~39,643 news articles from Mashable.com
    Features:
        58 attributes (e.g., number of images, keywords, NLP sentiment)
        Target: shares (number of times the article was shared)
    Preprocessing:
        Remove articles published <30 days before scraping (to avoid age bias)
        Normalize features
        Optional: Downsample for class imbalance
        
Methods Used
Feature Selection

    PCA (Principal Component Analysis)
    Mutual Information, Fisher Score
    Lasso / Ridge Regularization
    Backward Stepwise Selection

Regression Models

    Linear Regression
    Ridge & Lasso Regression
    Support Vector Regression (SVR)
    Kernel Partial Least Squares (KPLS)

Classification Models

    Logistic & Penalized Logistic Regression
    SVM (Linear & RBF)
    Random Forest (Best performance)
    LDA & QDA
    k-Nearest Neighbors (kNN)

NLP for Content Features

    Bag-of-Words with top 300 keywords
    Mutual Information for word selection
    Stopword removal and filtering

Visualizations

    Feature importance plots (Lasso, Random Forest)
    SVM performance over training size
    Confusion matrices
    Word clouds of top predictive keywords
    Distribution of article shares

    
Future Work

    Try N-grams and deep learning (e.g., BERT embeddings)
    Build a Streamlit app for real-time article prediction
    Explore fake news detection using similar features
    Use clustering to explore article topic trends

Main packages:

    scikit-learn
    pandas, numpy
    matplotlib, seaborn, plotly
    nltk or spacy (for NLP)
    jupyter
    
news-popularity-prediction/
│
├── data/                         # Dataset + any scraped content
│   └── OnlineNewsPopularity.csv
│
├── notebooks/                   # Jupyter notebooks for EDA, modeling, etc.
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Selection.ipynb
│   ├── 03_Regression_Models.ipynb
│   ├── 04_Classification_Models.ipynb
│   ├── 05_NLP_Content_Analysis.ipynb
│
├── src/                         # Python modules
│   ├── utils.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── plots/                       # Save charts for use in README
│
├── README.md
├── requirements.txt
└── LICENSE


