# TA9-BadDataBusters-ML-Frontend - Diabetes Prediction App

## Overview
This repository hosts a **Diabetes Prediction Web Application** designed to assess the risk of diabetes using a machine learning model. Given that November is **Diabetes Awareness Month**, this app serves as an essential tool to promote awareness and encourage early screening. Users can input various health metrics, and the app provides real-time feedback on diabetes risk based on a trained Random Forest model. 

The application is built using **Streamlit**, enabling an interactive web interface, and the model is trained with **scikit-learn**.

**Link:** https://baddatabusters-assignment-9.streamlit.app/ 
---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset and Model Training](#dataset-and-model-training)
3. [Key Features and Functionality](#key-features-and-functionality)
4. [Deployment and Hosting](#deployment-and-hosting)
5. [Technical Workflow](#technical-workflow)
6. [Usage](#usage)
7. [Considerations and Future Enhancements](#considerations-and-future-enhancements)

---

## Project Structure
The project directory is organized as follows:

```plaintext
.
├── app.py               # Streamlit app for the user interface
├── diabetes_model_advanced.pkl # Trained RandomForest model saved with joblib
├── requirements.txt     # Required packages for the app
├── README.md            # Documentation for the project
└── resources/           # Lottie animation JSON files and additional assets
```

## **Dataset and Model Training**

The model is trained on the popular **PIMA Indians Diabetes Dataset**, which includes various health-related metrics. You can find this dataset hosted on **Plotly’s GitHub repository**.

### **Key Features of the Model**

- **Random Forest Classifier**: A robust ensemble method that provides high accuracy by combining multiple decision trees.
- **Feature Engineering**: Includes interaction terms like `BMI_Age` and `Glucose_BloodPressure` to capture complex relationships among metrics.
- **Hyperparameter Tuning**: Optimized using `GridSearchCV` with cross-validation to enhance model performance.
- **Evaluation Metrics**: Accuracy, F1 score, and ROC AUC are used to assess the model's predictive power.

### **Model Training Pipeline**

#### **Data Preprocessing**:

- Missing values are imputed using **mean imputation**.
- All features are **standardized** to improve model training stability.

#### **Hyperparameter Tuning**:

- Performed with `GridSearchCV` over parameters such as the **number of estimators**, **maximum tree depth**, and **minimum samples split**.

#### **Model Serialization**:

- The best-performing model is saved as `diabetes_model_advanced.pkl` using **joblib** for efficient loading in the **Streamlit app**.

## **Key Features and Functionality**

The app offers an **intuitive interface** and **real-time predictions**, empowering users to assess diabetes risk based on personal metrics.

### **Interactive Input:**

- Users input metrics like **age**, **glucose level**, **blood pressure**, and **BMI** using **sliders**.

### **Real-time Risk Prediction:**

- Based on user inputs, the trained model evaluates **diabetes risk**.

### **Confidence Level Visualization:**

- A **gauge meter** provides a visual indicator of the **prediction confidence level**.

### **Health Tips and Recommendations:**

- Provides tailored advice based on the prediction, encouraging users to take **proactive steps in managing their health**.

### **Educational Expander Sections:**

- Each health metric is explained in detail, **educating users on its significance in diabetes risk**.

## **Deployment and Hosting**

The app is designed to be deployed on **Streamlit Cloud** or **Heroku**. Deployment instructions for **Streamlit Cloud** are provided below:

### **1. Create a Streamlit Cloud Account**
- Visit **Streamlit Cloud** and sign up.

### **2. Fork or Clone the Repository**
- Ensure your **GitHub repository** is public.

### **3. Deploy the App**
- In Streamlit Cloud, select **"Deploy an app"** and connect to your **GitHub repository**.

### **4. Specify the Entry File**
- Set `app.py` as the main file for the app.

Once deployed, the app is hosted on a **unique URL**, allowing users to access it easily.

## **Technical Workflow**

The following steps summarize the app's core workflow:

### **1. Data Collection**
- The user inputs **health metrics** through an interactive **UI**.

### **2. Preprocessing**
- The app prepares user inputs, handling **missing values** and **scaling**.

### **3. Prediction**
- The model makes a prediction based on the inputs, returning a **binary outcome** (**High** or **Low risk**) and a **confidence score**.

### **4. Feedback and Visualization**
   - **High Risk**: If the model detects a high risk, a **warning message** and **health tips** are displayed.
   - **Low Risk**: If the model detects low risk, a **success message** is shown, and **maintenance health tips** are provided.

### **5. Educational Content**
- The app provides **detailed explanations** for each health metric, helping users understand its significance.

## **Usage**

To use the app, follow these steps:

### **1. Enter Health Metrics**
- Input details such as:
  - **Number of pregnancies**
  - **Glucose level**
  - **Blood pressure**
  - **Skin thickness**
  - **Insulin level**
  - **BMI**
  - **Diabetes pedigree function**
  - **Age**

### **2. Interpret the Results**
- The app provides a **diabetes risk prediction** and displays the **confidence level** on a **gauge meter**.

### **3. Read Health Tips**
- Based on the risk level, users receive **actionable health tips** and **guidance**.

## **Considerations and Future Enhancements**

### **Clinical Utility**
- This model is a **preliminary tool**, and users should consult **healthcare providers** for a formal diagnosis.

### **Future Models**
- Incorporating additional health data (e.g., **lifestyle factors**) could improve **predictive accuracy**.

### **Data Security**
- As the app collects **sensitive health information**, deploying it with **encryption** and **user privacy features** will be a priority in future versions.

## **Conclusion**

The **Diabetes Prediction App** is a powerful, user-friendly tool created to promote **diabetes awareness** and encourage **proactive health management**. **Diabetes Awareness Month** highlights the importance of such resources, allowing individuals to assess their risk and learn more about **diabetes risk factors**. This app is an accessible entry point for understanding and mitigating the risks of diabetes.





