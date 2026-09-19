import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
import xgboost as xgb
import warnings
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

print("Загрузка данных...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

print("Предобработка данных...")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)
df_encoded.fillna(0, inplace=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Обучение моделей...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Метрики Логистической регрессии ---")
print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, lr_probs):.4f}")
print(classification_report(y_test, lr_preds, target_names=['Не ушел (0)', 'Ушел (1)']))

print("\n--- Метрики XGBoost ---")
print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, xgb_probs):.4f}")
print(classification_report(y_test, xgb_preds, target_names=['Не ушел (0)', 'Ушел (1)']))

print("Генерация визуализаций...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_display = ConfusionMatrixDisplay.from_estimator(
    xgb_model, X_test_scaled, y_test, display_labels=['Не ушел', 'Ушел'],
    cmap='Blues', ax=axes[0]
)
axes[0].set_title('Матрица ошибок (XGBoost)')

lr_display = RocCurveDisplay.from_estimator(lr_model, X_test_scaled, y_test, ax=axes[1], name='Logistic Regression')
lr_display.line_.set_color('blue')

xgb_display = RocCurveDisplay.from_estimator(xgb_model, X_test_scaled, y_test, ax=axes[1], name='XGBoost')
xgb_display.line_.set_color('red')

axes[1].plot([0, 1], [0, 1], 'k--', label='Random Guess')
axes[1].set_title('ROC-кривая (Сравнение моделей)')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()
print("Готово.")
