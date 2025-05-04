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
GAIT-BASED-CHECKIN-SYS/
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
|--- demo_assets/
     |--- demo.mov
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
|--- test_data/
    |---Test_Sample1.csv
    |---Test_Sample2.csv
    .
    .
    .
    |---Test_Sample11.csv    
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
pip install streamlit pandas numpy matplotlib seaborn joblib scipy
pip install scikit-learn==1.3.0
```
**Step 3:** Train the model (Random Forest Classifier)
```bash
python train.py
```
**Step 4:**  Model Training and Evaluation

- After running the training script, the terminal will display:
  - **Precision**, **Recall**, **F1-Score**, and **Support** for each class
  - **Overall Accuracy**
  - **Macro Avg** and **Weighted Avg** performance metrics
  - A **Confusion Matrix** visualizing the model’s predictions
- Once the output is displayed:
  - Open a **new terminal** in your code editor (e.g., **VS Code** or **PyCharm**) to continue with the next steps

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

### Demo:

A demo video of this application is attached in the demo_assets folder.

### Future Improvements:

- **Live Data Streaming from Mobile Devices**  
  Replace manual CSV uploads with real-time accelerometer data collection via a dedicated mobile interface.

- **Cloud-Based Deployment**  
  Host the application on a cloud platform to enable secure, remote access from any device.

- **Improved Prediction Accuracy**  
  Enhance the existing algorithm to better adapt to individual gait patterns and reduce false predictions.
