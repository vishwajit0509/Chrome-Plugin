import pandas as pd
import re
import logging

from src.logger import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import os

# Configure logger
from src.logger import configure_logger
configure_logger()
logger = logging.getLogger(__name__)


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# File paths
DATA_PATH = r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\notebooks\data.csv"
OUTPUT_PATH = r"C:\Users\Public\cleaned_data_final.csv"  

class DataPreprocessing:
    """Handles text preprocessing for NLP datasets."""

    def __init__(self, file_path, text_column):
        """
        Initialize with a file path and text column name.
        :param file_path: Path to the CSV file.
        :param text_column: Name of the column containing text data.
        """
        self.file_path = file_path
        self.text_column = text_column
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Load dataset
        self.df = self.load_data()

    def load_data(self):
        """Loads CSV data into a Pandas DataFrame."""
        try:
            logger.info(f"Loading dataset from {self.file_path}...")
            df = pd.read_csv(self.file_path)
            logger.info("Dataset loaded successfully!")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def clean_html(self, text):
        """Remove HTML tags from text."""
        return BeautifulSoup(text, "html.parser").get_text()

    def remove_special_chars(self, text):
        """Remove special characters and punctuation."""
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def tokenize(self, text):
        """Tokenize text into words."""
        return word_tokenize(text)

    def remove_stopwords(self, words):
        """Remove stopwords from tokenized words."""
        return [word for word in words if word not in self.stop_words]

    def lemmatize(self, words):
        """Lemmatize words to their root form."""
        return [self.lemmatizer.lemmatize(word) for word in words]

    def preprocess_text(self, text):
        """Apply all preprocessing steps to a given text."""
        if not isinstance(text, str):
            return ""

        text = self.clean_html(text)
        text = self.remove_special_chars(text)
        text = self.lowercase(text)
        words = self.tokenize(text)
        words = self.remove_stopwords(words)
        words = self.lemmatize(words)
        return " ".join(words)

    def process_data(self):
        """Apply preprocessing to the text column and handle missing values."""
        try:
            if self.text_column not in self.df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in dataset.")

            logger.info("Handling missing values...")
            self.df.dropna(subset=[self.text_column], inplace=True)

            # Print before preprocessing
            print("\n📌 Before Preprocessing:")
            print(self.df[[self.text_column]].head())

            logger.info("Applying text preprocessing...")
            self.df["cleaned_text"] = self.df[self.text_column].astype(str).apply(self.preprocess_text)

            # Print after preprocessing
            print("\n✅ After Preprocessing:")
            print(self.df[["cleaned_text"]].head())

            # Save cleaned data
            logger.info("Saving cleaned data...")
            self.df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
            logger.info(f"Cleaned data saved to {OUTPUT_PATH}")
            logger.info(f"File saved. Checking file existence: {os.path.exists(OUTPUT_PATH)}")

            return self.df

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

# Run preprocessing
if __name__ == "__main__":
    TEXT_COLUMN = "review" 

    if os.path.exists(DATA_PATH):
        preprocessor = DataPreprocessing(DATA_PATH, TEXT_COLUMN)
        cleaned_df = preprocessor.process_data()
        print("✅ Preprocessing completed. Cleaned data saved to 'cleaned_data_final.csv'.")
    else:
        logger.error(f"❌ File '{DATA_PATH}' not found!")
