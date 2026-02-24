# 🏦 Customer Churn Prediction

A machine learning web application that predicts the likelihood of a bank customer leaving (churning) based on various demographic and financial factors.

## 🚀 Live Demo
The app is designed to be deployed on [Streamlit Cloud](https://share.streamlit.io/).

## 🛠️ Features
- **Real-time Prediction**: Enter customer data and get instant churn probability.
- **Improved UI**: Clean, columns-based interface for easy data entry.
- **Robust Model**: Powered by an Artificial Neural Network (ANN) trained on the Churn Modelling dataset.
- **Scalable**: Uses pre-trained scikit-learn scaler for consistent input normalization.

## 📂 Project Structure
- `app.py`: The Streamlit web application.
- `model.h5`: Trained Keras Neural Network model.
- `scaler.pkl`: Serialized Scikit-Learn StandardScaler (Crucial for preprocessing).
- `Churn_Modelling.csv`: Raw dataset used for training/reference.
- `churn_prediction.ipynb`: Notebook containing EDA and model training logic.
- `requirements.txt`: Python dependencies.

## 💻 Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Roopendra-M/Customer_churn_Prediction.git
   cd Customer_churn_Prediction
   ```

2. **Create a virtual environment (Conda example)**:
   ```bash
   conda create -n churn python=3.10
   conda activate churn
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Why `scaler.pkl` is crucial?
The `scaler.pkl` file contains the `StandardScaler` used during model training. In Deep Learning, feature scaling (bringing values like Salary and Age to a similar range) is essential for convergence. 
**Without this file, the app would feed raw, unscaled numbers into the model, leading to completely incorrect predictions.**

## 📊 Dataset
The model is trained on the Kaggle Churn Modelling dataset, which includes features like:
- Credit Score
- Geography (France, Germany, Spain)
- Gender
- Age & Tenure
- Balance & Number of Products
- Member Activity Status
