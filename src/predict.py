import joblib
import pandas as pd
import numpy as np
from .feature_extraction import segment_and_extract

def predict_employee(model_path, uploaded_df, threshold=0.85):
    model = joblib.load(model_path)

    # Extract features
    accel_data = uploaded_df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']].values
    features = segment_and_extract(accel_data)

    # Predict for each segment
    preds = model.predict(features)
    unique, counts = np.unique(preds, return_counts=True)

    majority_class = unique[np.argmax(counts)]
    confidence = counts.max() / sum(counts)

    # Get actual model class name (e.g., RandomForestClassifier)
    model_type = type(model).__name__

    if confidence >= threshold:
        return majority_class, confidence, model_type
    else:
        return None, confidence, model_type

