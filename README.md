# CodeAlpha Data Science Internship Projects

Welcome to the official repository for the **CodeAlpha Data Science Internship**. This codebase contains production-grade Python implementations, Exploratory Data Analysis (EDA), machine learning regression pipelines, evaluation suites, interactive Jupyter Notebooks, and exported model binaries across 3 data science domain tasks.

---

## 📁 Repository Architecture

```text
CodeAlpha/
├── CodeAlpha_Unemployment Analysis with Python/                          # Task 1: Unemployment Analysis
│   ├── Unemployment in India.csv         # Primary rural vs urban dataset
│   ├── Unemployment_Rate_upto_11_2020.csv # Regional & time-series dataset (2020)
│   ├── unemployment_analysis.py          # Modular Python analysis script
│   ├── unemployment_analysis.ipynb       # Interactive Jupyter Notebook
│   └── outputs/                          # Generated plots & visualization charts
│       ├── correlation_heatmap.png
│       ├── rural_vs_urban_unemployment.png
│       ├── top_affected_states.png
│       └── unemployment_time_series_trends.png
│
├── CodeAlpha_Car Price Prediction with Machine Learning/                        # Task 2: Car Price Prediction
│   ├── car data.csv                      # Used car dataset
│   ├── car_price_prediction.py           # Machine learning training script
│   ├── car_price_prediction.ipynb        # Interactive Jupyter Notebook
│   └── outputs/                          # Serialized model & plots
│       ├── actual_vs_predicted_car_prices.png
│       ├── car_price_model.pkl           # Best model binary (Gradient Boosting)
│       └── feature_importance.png
│
├──CodeAlpha_Sales Prediction using Python/                      # Task 3: Sales Prediction
│   ├── Advertising.csv                   # Advertising spend dataset (TV, Radio, Newspaper)
│   ├── sales_prediction.py               # Regression modeling script
│   ├── sales_prediction.ipynb            # Interactive Jupyter Notebook
│   └── outputs/                          # Serialized model & plots
│       ├── actual_vs_predicted_sales.png
│       ├── feature_importance_sales.png
│       ├── sales_correlation_heatmap.png
│       └── sales_prediction_model.pkl    # Best model binary (Random Forest)
│
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git ignore rules
└── README.md                              # Main documentation
```

---

## 📊 Task Breakdowns & Results

### 1️⃣ Task 1: Unemployment Analysis in India
- **Objective**: Analyze the impact of COVID-19 lockdowns on unemployment rates in India, compare Rural vs. Urban trends, and identify the most impacted states.
- **Datasets**: `Unemployment in India.csv` and `Unemployment_Rate_upto_11_2020.csv`.
- **Key Preprocessing**:
  - Cleaned & normalized column headers (removed whitespace, converted to lowercase `snake_case`).
  - Handled null values and parsed date columns using `pd.to_datetime`.
- **Findings & Statistics**:
  - **Pre-Lockdown Average Unemployment Rate**: `9.23%`
  - **Lockdown Peak Average Unemployment Rate (March – June 2020)**: `16.74%` (a **+81.4% spike**).
  - **Most Affected States**: Haryana (`27.48%`), Tripura (`25.06%`), Jharkhand (`19.54%`), Bihar (`19.47%`), and Delhi (`18.41%`).
- **Visualizations**: Time-series line plot with highlighted COVID-19 lockdown window, Rural vs. Urban box plots, top 12 affected states bar chart, and metric correlation matrix.

---

### 2️⃣ Task 2: Car Price Prediction with Machine Learning
- **Objective**: Predict used car selling prices based on vehicle specifications and usage metrics.
- **Dataset**: `car data.csv`.
- **Feature Engineering & Encoding**:
  - Engineered `car_age = current_year - year`.
  - Applied One-Hot Encoding (`pd.get_dummies`) for `fuel_type`, `selling_type`, and `transmission`.
  - Scaled train/test split: 80% Training / 20% Testing.
- **Model Evaluation**:
  | Model | $R^2$ Score | MAE (Lakhs) | RMSE (Lakhs) |
  | :--- | :---: | :---: | :---: |
  | **Linear Regression** | 0.8489 | 1.2164 | 1.8658 |
  | **Random Forest Regressor** | 0.9595 | 0.6369 | 0.9664 |
  | **Gradient Boosting Regressor** | **0.9617** | **0.5899** | **0.9394** |
- **Exported Binary**: `car_price_model.pkl` (Gradient Boosting Regressor serialized using `joblib`).

---

### 3️⃣ Task 3: Sales Prediction using Python
- **Objective**: Predict sales revenue based on advertising expenditure across TV, Radio, and Newspaper channels.
- **Dataset**: `Advertising.csv`.
- **Correlation Analysis**:
  - **TV Spend Correlation**: `r = 0.782` (Primary sales driver).
  - **Radio Spend Correlation**: `r = 0.576`.
  - **Newspaper Spend Correlation**: `r = 0.228`.
- **Model Evaluation**:
  | Model | $R^2$ Score | MAE | RMSE |
  | :--- | :---: | :---: | :---: |
  | **Multi-Variable Linear Regression** | 0.8994 | 1.4608 | 1.7816 |
  | **Random Forest Regressor** | **0.9813** | **0.6201** | **0.7686** |
- **Exported Binary**: `sales_prediction_model.pkl` (Random Forest Regressor serialized using `joblib`).

---

## ⚡ Setup & Execution Guide

### Prerequisites
Ensure Python 3.9+ and `git` are installed on your machine.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/murali-172k6/codealpha_tasks.git
   cd codealpha_tasks
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Execution Commands

- **Run Task 1 (Unemployment Analysis)**:
  ```bash
  python unemployment/unemployment_analysis.py
  ```

- **Run Task 2 (Car Price Prediction)**:
  ```bash
  python "car prediction/car_price_prediction.py"
  ```

- **Run Task 3 (Sales Prediction)**:
  ```bash
  python "Sales Prediction/sales_prediction.py"
  ```

- **Run Interactive Jupyter Notebooks**:
  ```bash
  jupyter notebook
  ```

---

## 🧑‍💻 Author & Acknowledgments

- **Author**: Murali (`murali-172k6`)
- **Role**: Data Science Intern @ CodeAlpha
- **Acknowledgment**: Thanks to **CodeAlpha** for providing real-world datasets and internship domain tasks.
