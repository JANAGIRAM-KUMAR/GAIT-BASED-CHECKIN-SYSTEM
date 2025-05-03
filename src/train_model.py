

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.data_loader import load_all_employee_data

def train_model(data_dir, model_path):
    X, y = load_all_employee_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, class_weight='balanced')


    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test)))
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

"""


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.data_loader import load_all_employee_data

def train_model(data_dir, model_path):
    # Load data and labels
    X, y = load_all_employee_data(data_dir)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Gradient Boosting Classifier
    clf = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # Evaluate performance
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

"""

"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.data_loader import load_all_employee_data

def train_model(data_dir, model_path):
    # Load features and labels
    X, y = load_all_employee_data(data_dir)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize KNN Classifier
    clf = KNeighborsClassifier(
        n_neighbors=5,        # You can tune this (try odd values like 3, 5, 7)
        weights='distance',   # 'uniform' or 'distance'
        algorithm='auto'      # 'auto', 'ball_tree', 'kd_tree', 'brute'
    )

    # Train the model
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

"""