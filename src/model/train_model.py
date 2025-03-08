import os
import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File paths
FEATURES_PATH = r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\src\features\tfidf_features_final.csv"
LABELS_PATH = r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\src\data\cleaned_data_final.csv"
MODEL_DIR = r"C:\Users\vishw\OneDrive\Desktop\project2\Chrome-Plugin\models"


os.makedirs(MODEL_DIR, exist_ok=True)

class ModelTraining:
    """Handles training and evaluation of multiple classification models."""

    def __init__(self, features_path, labels_path, target_column):
        """
        Initialize model training class.
        :param features_path: Path to the TF-IDF feature file.
        :param labels_path: Path to the preprocessed data file containing labels.
        :param target_column: Name of the target column.
        """
        self.features_path = features_path
        self.labels_path = labels_path
        self.target_column = target_column

        
        self.X, self.y = self.load_data()

        # Models to train
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            "Support Vector Machine": SVC(kernel='linear', probability=True),
            "Naive Bayes": MultinomialNB()
        }

    def load_data(self):
        """Loads TF-IDF features and labels."""
        try:
            logging.info("Loading features and labels...")
            X = pd.read_csv(self.features_path)
            y = pd.read_csv(self.labels_path)[self.target_column]
            logging.info("Data loaded successfully!")
            return X, y
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance with multiple metrics."""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        
        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)

        
        report = classification_report(y_test, y_pred)

        return accuracy, precision, recall, f1, roc_auc, report

    def train_and_evaluate_models(self):
        """Train and evaluate multiple models, then save the best one."""
        try:
            logging.info("Splitting data into training and testing sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )

            best_model = None
            best_accuracy = 0

            
            results = []

            for model_name, model in self.models.items():
                logging.info(f"Training {model_name}...")
                model.fit(X_train, y_train)

               
                accuracy, precision, recall, f1, roc_auc, report = self.evaluate_model(model, X_test, y_test)

                # Log results
                logging.info(f"Model: {model_name}")
                logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
                if roc_auc is not None:
                    logging.info(f"ROC-AUC Score: {roc_auc:.4f}")
                logging.info(f"Classification Report for {model_name}:\n{report}")

                
                model_path = os.path.join(MODEL_DIR, f"{model_name.replace(' ', '_')}.pkl")
                joblib.dump(model, model_path)
                logging.info(f"Model saved at {model_path}")

                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = (model_name, model_path)

                
                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                    "ROC-AUC": roc_auc
                })

            
            logging.info("\n🔹 Final Model Comparison:")
            results_df = pd.DataFrame(results)
            logging.info(f"\n{results_df}")

            logging.info(f"✅ Best Model: {best_model[0]} | Saved at: {best_model[1]}")

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise


if __name__ == "__main__":
    TARGET_COLUMN = "sentiment"  

    if os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH):
        trainer = ModelTraining(FEATURES_PATH, LABELS_PATH, TARGET_COLUMN)
        trainer.train_and_evaluate_models()
        print("✅ Model training completed. Best model saved successfully!")
    else:
        logging.error("❌ Feature or label file not found!")
