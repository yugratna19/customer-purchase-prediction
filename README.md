# Customer Purchase Prediction

## Overview
This project predicts whether a customer will make a repeat purchase within the next 30 days using a synthetic dataset of customer transactions. It includes exploratory data analysis (EDA), simplified PySpark-based feature engineering, training three models (Logistic Regression, Random Forest, XGBoost), evaluation with F1-score and ROC-AUC, and deployment via FastAPI.

## Repository Structure
- `eda-for-predict-repeat-customer-purchases.ipynb`: Exploratory data analysis with visualizations of feature distributions and correlations.
- `modeling-for-predict-repeat-customer-purchases.ipynb`: Feature engineering, model training (Logistic Regression, Random Forest, XGBoost), evaluation, and visualizations (ROC curves, confusion matrices, feature importance).
- `etl.py`: PySpark script for data preprocessing and feature engineering (average order value, purchase frequency, preferred product category, recency).
- `train.py`: Script to train all three models, perform cross-validation, and save the best model based on F1-score.
- `app.py`: FastAPI script for serving the best model via an API.
- `customer_purchase_dataset.csv`: Input dataset (place in data folder).
- `best_model.pkl`: Saved best model.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yugratna19/customer-purchase-prediction.git
   cd customer-purchase-prediction
   ```

2. **Install Dependencies**:
   Create a virtual environment and install required packages:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Requirements File** (`requirements.txt`):
   In requirements.txt

4. **Run ETL**:
   Place the dataset (`customer_purchase_dataset.csv`) in the data folder, then run:
   ```bash
   python etl.py
   ```

5. **Run Training**:
   ```bash
   python train.py
   ```

6. **Run API Locally**:
   ```bash
   uvicorn deploy:app --reload
   ```

## Results
- **Models Trained**: Logistic Regression, Random Forest, XGBoost.
- **Evaluation Metrics**: 5-fold cross-validated F1-score and ROC-AUC on test set.
- **Best Model**: Selected based on highest CV F1-score (XGBoost).
- **Visualizations** (in `modeling.ipynb`):
  - ROC curves and confusion matrices for each model.
  - Feature importance chart for Random Forest and XGBoost.

## API Example
- **Method**: POST
- **Example Payload**:
  ```json
  {
   "total_amount": 75000.00,
   "product_category": "Clothing",
   "payment_type": "Mobile Payment",
   "delivery_location": "Kathmandu",
   "days_since_last_purchase": 2,
   "prev_purchases": 20,
   "total_spent_historical": 900000.00,
   "avg_order_value": 45000.00,
   "preferred_category": "Clothing"
}

- **Example Response**:
  ```json
  {
   "predicted_class": "Repeat Purchase",
   "probability_no_repeat": 0.45,
   "probability_repeat": 0.55
  }
  ```

## Notes
- **Feature Engineering**:
  - Recency: `days_since_last_purchase`
  - Frequency: `prev_purchases`
  - Monetary: `avg_order_value`
  - Preferred Category: `preferred_category` (most frequent prior category)
  - Outlier Handling: `total_amount` clipped at 99th percentile
- **Model Selection**: Best model chosen via 5-fold CV F1-score.
- **Scalable Processing**: PySpark used for preprocessing.
- **Deployment**: FastAPI serves the best model, saved as `.pkl`.
- For further improvements, hyperparameter tuning or additional data.


## View as Live API
You can access and test the deployed API directly via Hugging Face Spaces:

- Base URL: https://yugratna-customer-purchase-api.hf.space
- Swagger UI (Interactive Docs): https://yugratna-customer-purchase-api.hf.space/docs
