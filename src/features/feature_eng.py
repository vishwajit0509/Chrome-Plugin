#FEATURE ENGINEERING
import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
from src.logger import logging

#configure logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#file paths

INPUT_PATH = r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\cleaned_data_final.csv"
FEATURES_OUTPUT_PATH=r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\tfidf_features_final.csv"

class FeatureEngineering:
    """Handles feature engineering for NLP datasets."""
    def __init__(self,input_path,text_column):
        """
        Initialize with file path and text column.
        :param input_path:path of preprocessed CSV file.
        :param text_column: Column containing text data.
        """

        self.input_path = input_path
        self.text_column = text_column
        self.df = self.load_data()
        self.vectorizer = TfidfVectorizer ( max_features=1000,
                                        stop_words="english",
                                        ngram_range=(1,2),
                                        lowercase=True,
                                        strip_accents="unicode",
                                        min_df=5, 
                                        max_df=0.9  
                                        )

    def load_data(self):

        """load the preprocessed data"""
        try:
            logging.info(f"Loading dataset from {self.input_path}...")
            df = pd.read_csv(self.input_path)
            logging.info("Dataset loaded successfully!")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def apply_tfidf(self):
        """Apply TF-IDF to text data."""
        try:
            if self.text_column not in self.df.columns:
                raise ValueError(f"Column '{self.text_column}' not found in dataset.")
            logging.info("Applying TF-IDF tranformation...")
            X_tfidf = self.vectorizer.fit_transform(self.df[self.text_column])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)

            logging.info("TF-IDF transformation applied successfully!")
            return tfidf_df
        except Exception as e:
            logging.error(f"Error in TF-IDF processing: {e}")
            raise

    def save_features(self, tfidf_df):

        """Save TF-IDF features to a CSV file."""
         
        try:
            os.makedirs(os.path.dirname(FEATURES_OUTPUT_PATH), exist_ok=True)
            tfidf_df.to_csv(FEATURES_OUTPUT_PATH, index=False, encoding="utf-8")
            logging.info(f"TF-IDF features saved to {FEATURES_OUTPUT_PATH}")
        except Exception as e:
            logging.error(f"Error saving TF-IDF features: {e}")
            raise

if __name__ == "__main__":
    TEXT_COLUMN = "cleaned_text"
    
    if os.path.exists(INPUT_PATH):
        fe = FeatureEngineering(INPUT_PATH, TEXT_COLUMN)
        tfidf_features = fe.apply_tfidf()
        fe.save_features(tfidf_features)
        print("✅ TF-IDF feature engineering completed. Features saved successfully.")
    else:
        logging.error(f"❌ Preprocessed data file '{INPUT_PATH}' not found!")

        



        
       






