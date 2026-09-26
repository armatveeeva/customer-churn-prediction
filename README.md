# Прогнозирование оттока клиентов телекоммуникационной компании 

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange)](https://shap.readthedocs.io/)

Полноценный ML-проект, решающий бизнес-задачу прогнозирования оттока клиентов телекоммуникационной компании. Репозиторий содержит полный пайплайн: от разведочного анализа (EDA) и продвинутого создания признаков до обучения моделей с кросс-валидацией и интерпретации результатов с помощью SHAP. Финальным этапом является деплой интерактивного веб-приложения на Streamlit.

## Описание предметной области

В телекоммуникационной индустрии удержание существующих клиентов обходится компании в 5–25 раз дешевле, чем привлечение новых. Прогнозирование оттока позволяет выявить абонентов с высокой вероятностью расторжения контракта и предложить им персонализированные условия удержания.

Данный проект реализует не только предиктивную модель, но и слой интерпретируемости. Использование алгоритма SHAP позволяет перейти от модели «черного ящика» к прозрачному анализу: определить, какие именно факторы вносят наибольший вклад в прогноз для каждого конкретного клиента, обеспечивая бизнес обоснованными данными для принятия решений.

## Технологический стек

- **Язык:** Python 3.11+
- **Обработка данных:** Pandas, NumPy
- **Машинное обучение:** Scikit-learn (кросс-валидация, метрики, пайплайны)
- **Алгоритмы:** Logistic Regression, Random Forest, XGBoost
- **Объяснимый ИИ:** SHAP (SHapley Additive exPlanations)
- **Визуализация:** Matplotlib, Seaborn
- **Деплой и UI:** Streamlit

## Быстрый старт

### Предварительные требования

- Python 3.11 или выше
- Менеджер пакетов pip

### Установка и настройка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/armatveeeva/customer-churn-prediction.git
cd customer-churn-prediction
