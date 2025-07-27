# employee-salary-prediction_Krushi-Koyagura

# 💼 Employee Salary Prediction

This project predicts whether an individual's salary is above or below ₹50K using classification models trained on census-like data.

## 🔍 Project Highlights

- Model Comparison: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, SVC
- Accuracy Metrics Visualization
- Streamlit App Interface
- Label Encoding for all categorical features
- Modular, clean pipeline

## 🧠 Technologies Used

- Python, Pandas, Scikit-learn
- Matplotlib & Seaborn for visualizations
- Streamlit for UI
- Joblib for model saving

## 📁 Folder Structure

```
Employee-Salary-Prediction/
│
├── app.py                      # Streamlit app
├── best_model.pkl              # Final trained model
├── feature_order.pkl           # Column order for prediction
├── target_encoder.pkl          # Encoder for output label
├── adult 3.csv                 # Input dataset
├── requirements.txt            # Package dependencies
├── Employee-Salary-Prediction.ipynb  # Jupyter notebook
├── encoders/                   # Encoded categorical features
│   ├── education_encoder.pkl
│   ├── occupation_encoder.pkl
│   └── ...
```

## 🚀 How to Run

### 📌 Step 1: Clone the repository
```bash
git clone https://github.com/your-username/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
```

### 📌 Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### 📌 Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

## 👤 Author

Krushi Koyagura — _Machine Learning Enthusiast_
