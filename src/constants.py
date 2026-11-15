DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

COLUMN_NAMES_RU = {
    "gender": "Пол",
    "SeniorCitizen": "Пенсионер",
    "Partner": "Партнер",
    "Dependents": "Иждивенцы",
    "tenure": "Срок обслуживания",
    "PhoneService": "Телефония",
    "MultipleLines": "Несколько линий",
    "InternetService": "Тип интернета",
    "OnlineSecurity": "Онлайн-защита",
    "OnlineBackup": "Онлайн-бекап",
    "DeviceProtection": "Защита устройств",
    "TechSupport": "Техподдержка",
    "StreamingTV": "Стриминг ТВ",
    "StreamingMovies": "Стриминг кино",
    "Contract": "Контракт",
    "PaperlessBilling": "Электронный счет",
    "PaymentMethod": "Способ оплаты",
    "MonthlyCharges": "Ежемесячный платеж",
    "TotalCharges": "Общий платеж",
    "Churn": "Отток",
}

VALUE_MAPS = {
    "Пол": {"Male": "Мужской", "Female": "Женский"},
    "Пенсионер": {1: "Да", 0: "Нет"},
    "Партнер": {"Yes": "Да", "No": "Нет"},
    "Иждивенцы": {"Yes": "Да", "No": "Нет"},
    "Телефония": {"Yes": "Да", "No": "Нет"},
    "Несколько линий": {"Yes": "Да", "No": "Нет", "No phone service": "Нет телефона"},
    "Тип интернета": {"DSL": "DSL", "Fiber optic": "Оптоволокно", "No": "Нет"},
    "Онлайн-защита": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Онлайн-бекап": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Защита устройств": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Техподдержка": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Стриминг ТВ": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Стриминг кино": {"Yes": "Да", "No": "Нет", "No internet service": "Нет интернета"},
    "Контракт": {"Month-to-month": "Помесячный", "One year": "1 год", "Two year": "2 года"},
    "Электронный счет": {"Yes": "Да", "No": "Нет"},
    "Способ оплаты": {
        "Electronic check": "Электронный чек",
        "Mailed check": "Чек почтой",
        "Bank transfer (automatic)": "Банковский перевод",
        "Credit card (automatic)": "Кредитная карта",
    },
    "Отток": {"Yes": "Ушел", "No": "Остался"},
}

EDA_COLORS = {"success": "#4ECDC4", "danger": "#FF6B6B", "neutral": "#95A5A6"}

FE_COLORS = {"primary": "#2E86AB", "secondary": "#A23B72", "accent": "#F18F01"}

SERVICE_COLS = [
    "Онлайн-защита",
    "Онлайн-бекап",
    "Защита устройств",
    "Техподдержка",
    "Стриминг ТВ",
    "Стриминг кино",
]
