from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_all_employee_data


def train_model(data_dir, model_path):
    X, y = load_all_employee_data(data_dir)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)


    print(f"Number of samples: {len(X)}")
    print(f"Number of features per sample: {X.shape[1]}")

    if X.shape[1] != 19:
        raise ValueError(f"❌ Expected 19 features, but got {X.shape[1]}. Fix feature_extraction.py.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")
