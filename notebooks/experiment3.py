import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting' , True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , classification_report , precision_score , recall_score , f1_score

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore' , UserWarning)


MLFLOW_TRACKING_URI = 'https://dagshub.com/ay747283/Chrome-Plugin.mlflow'
dagshub.init(repo_owner='ay747283', repo_name='Chrome-Plugin' , mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('Logistic Reg Hyperparameter Tuning')


def preprocess_text(text):
    """Applies multiple text preprocessing steps."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopwords removal
    
    return text.strip()


def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    
    # Apply text preprocessing
    df["review"] = df["review"].astype(str).apply(preprocess_text)
    
    # Filter for binary classification
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer



def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")



if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)