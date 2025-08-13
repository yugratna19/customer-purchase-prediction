import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib

def train_models(input_path):
    # Load processed data
    df = pd.read_csv(input_path)
    
    # Data preparation
    X = df.drop(['repeat_purchase_next_30_days', 'customer_id', 'order_date'], axis=1)
    y = df['repeat_purchase_next_30_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_features = ['total_amount', 'days_since_last_purchase', 'prev_purchases', 'avg_order_value']
    categorical_features = ['product_category', 'payment_type', 'delivery_location', 'preferred_category']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    
    # Models
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))])
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, eval_metric='auc'))])
    
    models = [
        (lr_pipeline, "Logistic Regression"),
        (rf_pipeline, "Random Forest"),
        (xgb_pipeline, "XGBoost")
    ]
    
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for pipeline, name in models:
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1').mean()
        pipeline.fit(X_train, y_train)
        test_f1 = f1_score(y_test, pipeline.predict(X_test))
        print(f"{name} CV F1: {cv_f1:.4f}, Test F1: {test_f1:.4f}")
        if cv_f1 > best_f1:
            best_f1 = cv_f1
            best_model = pipeline
            best_name = name
    
    # Save best model
    joblib.dump(best_model, 'best_model.pkl')
    print(f"Best Model: {best_name} with CV F1: {best_f1:.4f}")

if __name__ == "__main__":
    train_models(r'data\processed_data.csv')
