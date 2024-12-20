# Diabetes Risk Prediction Model

This project utilizes machine learning to predict the risk of diabetes based on various health parameters. The model is trained on a dataset that includes features such as age, gender, and various health symptoms. The app built using Streamlit allows users to input their data and receive a prediction regarding their diabetes risk.

## Table of Contents
- [Project Description](#project-description)
- [Technologies Used](#technologies-used)
- [Model Training](#model-training)
- [How to Run the App](#how-to-run-the-app)
- [Features](#features)
- [Model Evaluation](#model-evaluation)
- [Recommendations](#recommendations)

## Project Description
This project predicts whether a person is at risk of diabetes based on their health conditions, using a variety of machine learning algorithms including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) with Linear and RBF kernels
- Random Forest
- Naive Bayes
- Decision Tree

The app allows users to input health-related details, which are then processed to provide a risk assessment for diabetes.

## Technologies Used
- **Python**: The main programming language for model building and app development.
- **Streamlit**: A framework used for building the web app.
- **Scikit-learn**: For model training, evaluation, and preprocessing.
- **Pickle**: For saving and loading the trained model.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical computing and array handling.

## Model Training
The model was trained on a dataset containing information about various health parameters. The following classifiers were used:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Kernel SVM**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**

After training, the models were evaluated using accuracy metrics, and the **Random Forest** model showed the highest accuracy and was selected for deployment.

## How to Run the App

### Prerequisites:
- Python 3.7 or higher
- Install the required dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

### Running the App:
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/katakampranav/Diabetes-Prediction-Model.git
   ```

2. Install the necessary packages (listed in `requirements.txt`).

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Features
- **User Inputs**: Users can enter details like age, gender, and health-related symptoms (e.g., polyuria, sudden weight loss).
- **Prediction**: After submitting the form, the app will display whether the person is at risk of diabetes.
- **Real-time Results**: Predictions are displayed immediately based on the input data.

## Output Images
![Screenshot 2024-12-14 155753](https://github.com/user-attachments/assets/dbc03ca5-2563-41a4-ad39-ab04a56a4884)
![image](https://github.com/user-attachments/assets/28430da6-7c46-4956-892d-4f521ac5ac9f)
![image](https://github.com/user-attachments/assets/0c3ede38-3265-4181-9343-1c71196f31ae)

## Model Evaluation
The models were evaluated using accuracy metrics, and the results are summarized below:

| Classifier             | Accuracy   |
|------------------------|------------|
| Logistic Regression     | 90.20%     |
| KNN                    | 76.47%     |
| SVM (Linear Kernel)     | 86.27%     |
| SVM (RBF Kernel)        | 88.24%     |
| Naive Bayes            | 88.24%     |
| Decision Tree          | 78.43%     |
| Random Forest          | 96.08%     |

### Best Model: **Random Forest** with an accuracy of **96.08%**.

## Recommendations
Based on the accuracy results:
- **Random Forest** is the best model and is recommended for deployment.
- **SVM (RBF Kernel)** and **Naive Bayes** are also strong contenders.
- **KNN** and **Decision Tree** show lower performance and are not recommended for production.

## Author

This Diabetes Risk Prediction Model was developed by:
- [@katakampranav](https://github.com/katakampranav)
- Repository: [https://github.com/katakampranav/Diabetes-Prediction-Model](https://github.com/katakampranav/Diabetes-Prediction-Model)

## Feedback

For any feedback or queries, please reach out to me at katakampranavshankar@gmail.com.
