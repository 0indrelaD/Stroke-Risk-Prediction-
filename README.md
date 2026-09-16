# Stroke Risk Prediction using Machine Learning

A complete end-to-end data science project that explores a healthcare dataset and builds machine learning models to predict the likelihood of a patient having a stroke, based on demographic and health-related features.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Models Used](#models-used)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🩺 Overview

Stroke is one of the leading causes of death and disability worldwide. Early identification of high-risk individuals can enable timely intervention and significantly improve patient outcomes. This project uses supervised machine learning to classify whether a patient is likely to experience a stroke based on features such as age, hypertension, heart disease, average glucose level, BMI, smoking status, and more.

The notebook walks through the full data science lifecycle:

1. Data loading and inspection
2. Data cleaning and handling missing values
3. Exploratory Data Analysis (EDA) with visualizations
4. Feature encoding and scaling
5. Model building, training, and evaluation
6. Addressing class imbalance to boost model performance

---

## 📊 Dataset

The project uses a healthcare stroke prediction dataset (`q_dataset.csv`) containing patient-level records with the following features:

| Feature | Description |
|---|---|
| `id` | Unique patient identifier |
| `gender` | Male / Female / Other |
| `age` | Age of the patient |
| `hypertension` | 0 = No, 1 = Yes |
| `heart_disease` | 0 = No, 1 = Yes |
| `ever_married` | Yes / No |
| `work_type` | Type of occupation |
| `Residence_type` | Urban / Rural |
| `avg_glucose_level` | Average glucose level in blood |
| `bmi` | Body Mass Index |
| `smoking_status` | Smoking history |
| `stroke` | Target variable — 1 = had a stroke, 0 = did not |

> **Note:** The dataset is imbalanced, with far fewer positive (stroke = 1) cases than negative ones, which is addressed later in the pipeline.

---

## 🔄 Project Workflow

### 1. Data Loading & Inspection
- Loaded the dataset into a pandas DataFrame.
- Inspected shape, data types, and summary statistics using `.shape`, `.dtypes`, `.info()`, and `.describe()`.

### 2. Missing Value Treatment
- Identified missing values (primarily in the `bmi` column) using `.isnull().sum()`.
- Imputed missing numeric values using the **median** (more robust to outliers than the mean) and added a binary "was missing" indicator column for traceability.

---

## 📈 Exploratory Data Analysis

Extensive EDA was performed using **Matplotlib**, **Seaborn**, and **Plotly** to understand feature distributions and their relationship with stroke occurrence:

- **Age distribution** — histogram with KDE overlay
- **Age vs. Average Glucose Level** — scatter plot colored by stroke outcome
- **BMI by Stroke Status** — boxplot comparison
- **Stroke frequency by Gender, Hypertension, Heart Disease, Smoking Status, Work Type, Residence Type, and Marital Status** — bar/crosstab visualizations
- **Interactive Plotly histograms** — stroke frequency segmented by age and gender
- **Correlation heatmap** — relationships between all numeric features
- **Outlier detection** — boxplots across all features, with BMI outliers (> 47) removed

---

## 🛠️ Data Preprocessing

- **Categorical Encoding:** Applied `LabelEncoder` to convert categorical columns (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) into numeric form.
- **BMI Categorization:** Bucketed BMI into `Underweight`, `Normal Weight`, `Overweight`, and `Obese` for additional exploratory insight.
- **Feature Scaling:** Standardized features using `StandardScaler` (initial models) and `MinMaxScaler` (refined pipeline).
- **Outlier Removal:** Dropped extreme BMI outliers to reduce noise.
- **Train/Test Split:** Used `train_test_split` with stratification on the target variable to preserve class ratios.

---

## ⚖️ Handling Class Imbalance

Since stroke cases represent a small minority of the dataset, models trained on raw data tend to be biased toward predicting "no stroke." To address this, the project applies:

- **SMOTE (Synthetic Minority Over-sampling Technique)** from `imblearn` to oversample the minority class and create a balanced training set — a key step in the bonus task aimed at improving model accuracy by at least 10%.

---

## 🤖 Models Used

Multiple classification algorithms were trained and evaluated for comparison:

| Model | Library |
|---|---|
| Decision Tree Classifier | `sklearn.tree` |
| Random Forest Classifier | `sklearn.ensemble` |
| Logistic Regression | `sklearn.linear_model` |
| Support Vector Machine (SVM) | `sklearn.svm` |
| Naive Bayes (Gaussian) | `sklearn.naive_bayes` |
| K-Nearest Neighbors (KNN) | `sklearn.neighbors` |

Each model was evaluated using:
- **Accuracy Score**
- **Confusion Matrix** (visualized with Seaborn heatmaps)
- **Classification Report** (precision, recall, F1-score)

The trained scaler was also serialized using `pickle` for potential reuse in deployment.

---

## 📌 Results

- Baseline models were trained on the original (imbalanced) dataset and compared by accuracy in a summary DataFrame.
- After applying SMOTE oversampling and re-training Logistic Regression, KNN, and Random Forest, model performance on the minority (stroke) class improved substantially, as shown in the updated classification reports and confusion matrices.
- Feature correlation analysis confirmed **age**, **average glucose level**, **hypertension**, and **heart disease** as the most influential predictors of stroke risk.

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Data Handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** scikit-learn
- **Class Imbalance Handling:** imbalanced-learn (SMOTE)
- **Model Persistence:** pickle

---

## 📁 Project Structure

```
stroke-prediction/
│
├── q_dataset.csv                 # Raw dataset
├── stroke_prediction.ipynb       # Main analysis & modeling notebook
├── lr.pkl                        # Pickled StandardScaler object
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly imbalanced-learn
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/stroke-prediction.git
   cd stroke-prediction
   ```
2. Place `q_dataset.csv` in the project directory.
3. Open and run the notebook:
   ```bash
   jupyter notebook stroke_prediction.ipynb
   ```

---

## 💡 Key Insights

- Stroke risk increases noticeably with **age** — older patients show a much higher incidence.
- **Average glucose level** trends upward with age and correlates with higher stroke occurrence.
- Patients with **hypertension** and **heart disease** show a disproportionately higher stroke frequency.
- **BMI alone** is not a strong standalone predictor — its relationship with stroke shows high variance.
- The dataset's **class imbalance** significantly affects naive model performance, making resampling techniques like SMOTE essential for building a reliable classifier.

---

## 🔮 Future Improvements

- Hyperparameter tuning via `GridSearchCV` for all models.
- Cross-validation for more robust performance estimates.
- Feature engineering (e.g., interaction terms between age, glucose, and BMI).
- Model explainability using SHAP or LIME to interpret individual predictions.
- Deployment as a simple web app (Flask/Streamlit) for interactive risk prediction.

---

## 📄 License

This project is open-source and available for educational and research purposes. Please cite appropriately if reused.
