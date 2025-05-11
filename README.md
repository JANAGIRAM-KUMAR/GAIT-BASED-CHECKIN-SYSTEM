# 🏤 Stark Industries: Gait-Based Security System

### Problem Statement:

At Stark Industries, employees currently use physical keycards to check in and enter the building. As a security analyst, you have proposed a **contactless employee check-in system** that leverages the smartphone's built-in sensors (like the accelerometer) and **machine learning** to identify employees based on their **gait patterns**.

---

### Solution:

So, here’s the idea: What if employees could be identified just by the way they walk?

This project is a prototype for a **contactless employee check-in system** using **gait analysis**. The system uses data from the **accelerometer sensor** in a person’s smartphone. When someone walks into the building, their phone sends motion data to a server, which then checks if the walking pattern (gait) matches any registered employee. If there’s a match, access is granted — no cards or buttons required.

---

### Technologies Used:

- **Python** for backend logic and machine learning
- **Random Forest Classifier** (via `scikit-learn`) for gait recognition
- **Streamlit** for the web frontend interface
- **Joblib** to save and load trained models
- **Pandas, NumPy, SciPy** for data preprocessing
- **Matplotlib & Seaborn** for visualization
- **Physics Toolbox Sensor Suite** (Android app) for collecting accelerometer data

---

### Project Structure:
```
GAIT-BASED-CHECKIN-SYSTEM/
|
|--- app/
     |--- main.py
|--- data/
     |--- employee1/
          |--- empdata1A.csv
          |--- empdata1B.csv
          |--- empdata1C.csv
     |--- employee2/
          |--- empdata2A.csv
          |--- empdata2B.csv
          |--- empdata2C.csv
          |--- empdata2D.csv
          |--- empdata2E.csv
      .
      .
      .
     |--- employee30/
          |--- empdata30A.csv
          |--- empdata30B.csv
          |--- empdata30C.csv
|--- data_visualization/
     |--- emp_data.csv
|--- img/
     |--- background.jpg
|--- models/
     |--- gait_classifier.pkl
|--- src/
     |--- __init__.py
     |--- data_loader.py
     |--- feauture_extraction.py
     |--- predict.py
     |--- train_model.py
     |--- visualize_processing_pipeline.py
|--- testing_data/
    |---testdata1.csv
    |---testdata2.csv
    .
    .
    .
    |---testdata20.csv    
|--- .gitignore
|--- README.md
|--- train.py
```
---

### How to run this Project:

**Step 1:** Clone the repository
```bash
git clone https://github.com/JANAGIRAM-KUMAR/GAIT-BASED-CHECKIN-SYSTEM.git
cd GAIT-BASED-CHECKIN-SYSTEM
```

**Step 2:** Install required Packages
```bash
pip install --upgrade pip
pip install streamlit pandas numpy joblib matplotlib seaborn scikit-learn
```
**Step 3:** Train the model (XGBClassifier)
```bash
python train.py
```
**Step 4:**  Model Training and Evaluation

After preparing and filtering the accelerometer data using a high-pass filter, we trained a model using the XGBoost classifier.

Model Used: XGBClassifier

Windowing: Each CSV file was segmented into overlapping windows.
Features: Extracted from the filtered accelerometer signals (gravity removed).
Training Data: Aggregated from all .csv files under the data/ directory.
Once training is complete, the script outputs:

Classification Report:
```
Number of samples: 131360
Number of features per sample: 19

Accuracy: 0.8847
              precision    recall  f1-score   support

           0       0.99      0.89      0.94       183
           1       0.84      0.80      0.82      5141
           2       0.98      0.98      0.98       512
           3       0.88      0.88      0.88      3058
           4       0.95      0.94      0.95       701
           5       0.98      0.98      0.98       241
           6       0.97      0.96      0.96       175
           7       0.93      0.96      0.95       370
           8       0.95      0.91      0.93       176
           9       0.86      0.95      0.90      3736
          10       0.95      0.97      0.96       318
          11       0.84      0.84      0.84      6184
          12       0.93      0.92      0.93       558
          13       0.96      0.94      0.95       230
          14       1.00      0.95      0.97       273
          15       0.95      0.97      0.96       459
          16       0.96      0.90      0.93       122
          17       0.96      0.94      0.95       306
          18       0.97      0.96      0.96       381
          19       0.98      0.94      0.96       616
          20       0.96      0.98      0.97       163
          21       0.96      0.93      0.95       412
          22       0.99      0.94      0.97       265
          23       0.98      0.84      0.91       437
          24       0.96      0.92      0.94       322
          25       0.98      0.98      0.98       174
          26       0.99      0.91      0.95       150
          27       0.97      0.95      0.96       320
          28       0.97      0.99      0.98       184
          29       0.98      0.92      0.95       105

    accuracy                           0.88     26272
   macro avg       0.95      0.93      0.94     26272
weighted avg       0.89      0.88      0.88     26272

Model saved to models/gait_classifier.pkl
2025-05-11 11:55:01.033 python[26831:731172] +[IMKClient subclass]: chose IMKClient_Modern
2025-05-11 11:55:01.033 python[26831:731172] +[IMKInputSession subclass]: chose IMKInputSession_Modern
Segments Extracted: 120
Features per Segment: 19
```
**Precision** – Model’s ability to avoid false positives
**Recall** – Model’s ability to capture all actual positives
**F1-Score** – Harmonic mean of precision and recall
**Support** – Number of true samples per class

**Sensor Processing Visualization:**

This figure shows raw, filtered, and **gravity** components for each axis (X, Y, Z), clearly illustrating the effect of high-pass filtering (cutoff = 0.3 Hz):



**Step 5:** Run the Streamlit app
```bash
streamlit run app/main.py
```
**Step 6:** Live Application

Now you can now view your Streamlit app in your browser.
```
Local URL: http://localhost:8501
```

### Streamlit App Usage for Gait-Based Access Control:

- In the Streamlit app, you will:
  - **Upload accelerometer data CSV files** (i.e., test files) from the `test_data/` folder.
  - These CSV files simulate data captured directly from a mobile device as per the problem statement.
- After uploading:
  - The app will process the uploaded file and **predict whether to grant or deny access** for the employee.
  - Prediction is based on the trained gait recognition model using accelerometer data.
- Note:
  - Currently, the app uses a **file upload port** to simulate real-time data capture from a mobile device.
  - In future iterations, this can be replaced with live streaming or direct mobile app integration.

### Future Improvements:

- **Live Data Streaming from Mobile Devices**  
  Replace manual CSV uploads with real-time accelerometer data collection via a dedicated mobile interface.

- **Cloud-Based Deployment**  
  Host the application on a cloud platform to enable secure, remote access from any device.

- **Improved Prediction Accuracy**  
  Enhance the existing algorithm to better adapt to individual gait patterns and reduce false predictions.
