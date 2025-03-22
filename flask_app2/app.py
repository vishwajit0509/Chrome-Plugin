from flask import Flask, request, jsonify, render_template
import mlflow
import pickle
import os
import pandas as pd
import re
import string
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask_cors import CORS  # To avoid CORS issues with Chrome extension

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -----------------------------
# FLASK APP INITIALIZATION
# -----------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for your Chrome extension requests

# -----------------------------
# TEXT PREPROCESSING FUNCTIONS
# (Copied from your first working code)
# -----------------------------
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    """
    Apply all your preprocessing in the same order 
    as your first working code.
    """
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# -----------------------------
# YOUTUBE COMMENT FETCH
# -----------------------------
def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL (both 'youtube.com/watch' and 'youtu.be' formats).
    """
    parsed = urlparse(url)
    if "youtube" in parsed.netloc:
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None

def fetch_youtube_comments(video_url, api_key):
    """
    Fetch top-level YouTube comments using the YouTube Data API.
    Returns a list of raw comment strings.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: could not extract video ID")

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        for item in response.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# -----------------------------
# MLflow MODEL LOADING
# -----------------------------
mlflow.set_tracking_uri("https://dagshub.com/ay747283/Chrome-Plugin.mlflow")

model_name = "my_model"
# If your staging version is correct, keep it as is
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["staging"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{model_version}"
print(f"Fetching model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# IMPORTANT: Ensure this is the SAME vectorizer you used when training
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route("/")
def index():
    """
    For quick testing in the browser
    """
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON:
      {
        "video_url": "https://www.youtube.com/watch?v=XXXX"
      }
    Fetches comments, preprocesses them with the EXACT same pipeline,
    vectorizes them with the EXACT same vectorizer, 
    and uses your logistic regression model (which expects 20 features).
    """
    data = request.get_json()
    video_url = data.get("video_url")
    if not video_url:
        return jsonify({"error": "No video_url provided"}), 400

    # Provide your YouTube Data API key
    api_key = "AIzaSyAkd2c7rnaGcGLTUOE_wD9I0Y4sgNkJaPo"

    try:
        # 1) Fetch comments
        comments = fetch_youtube_comments(video_url, api_key)
        if not comments:
            return jsonify({"error": "No comments found"}), 404

        # 2) Preprocess EXACTLY like your first code
        processed_comments = [normalize_text(c) for c in comments]

        # 3) Vectorize
        X = vectorizer.transform(processed_comments)

        # 4) Predict with your logistic regression
        # If your model expects 20 features, X.shape[1] should be 20
        # If X.shape[1] != 20, it means there's still a mismatch in vectorizer
        predictions = model.predict(X)
        predictions = predictions.tolist()

        # Suppose your model uses 1 = positive, 0 = negative
        positive_count = predictions.count(1)
        negative_count = predictions.count(0)

        response_data = {
            "video_url": video_url,
            "total_comments": len(comments),
            "positive_comments": positive_count,
            "negative_comments": negative_count
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)
