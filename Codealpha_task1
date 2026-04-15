import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample dataset creation (replace with real data)
np.random.seed(42)
n_samples = 5000

data = {
    'income': np.random.normal(50000, 20000, n_samples),
    'debt': np.random.normal(20000, 10000, n_samples),
    'payment_history': np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.2, 0.3, 0.4]),  # 0=poor, 3=excellent
    'credit_age_months': np.random.randint(6, 360, n_samples),
    'num_defaults': np.random.poisson(0.5, n_samples),
    'credit_utilization': np.random.uniform(0, 1, n_samples),
    'num_credit_cards': np.random.randint(1, 10, n_samples)
}
df = pd.DataFrame(data)

# Feature engineering
df['debt_to_income'] = df['debt'] / df['credit_utilization'].clip(lower=0.01)
df['payment_score'] = df['payment_history'] * df['credit_age_months'] / 12
df['risk_score'] = (df['num_defaults'] * 20 + df['credit_utilization'] * 30 + 
                     (3 - df['payment_history']) * 15)

# Create target (creditworthy = 1, not = 0)
df['creditworthy'] = (df['risk_score'] < 50).astype(int)

# Prepare features
feature_cols = ['income', 'debt', 'payment_history', 'credit_age_months', 
                'num_defaults', 'credit_utilization', 'num_credit_cards',
                'debt_to_income', 'payment_score']
X = df[feature_cols]
y = df['creditworthy']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5)
}

# Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n{name}")
    print(classification_report(y_test, y_pred))

# Summary
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
results_df = pd.DataFrame(results).T
print(results_df)

# Cross-validation
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100), X_scaled, y, cv=5)
print(f"\nRandom Forest CV Score (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
