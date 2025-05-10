from src.train_model import train_model
from src.visualize_processing_pipeline import visualize_single_csv

train_model("data", "models/gait_classifier.pkl")
visualize_single_csv("data_visualization/emp_data.csv")

