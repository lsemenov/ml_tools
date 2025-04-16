#!/usr/bin/env python
import os
from clearml import Task, Logger

import numpy as np
from getpass import getpass

from sklearn.model_selection import train_test_split
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from catboost import CatBoostClassifier

# Глобальный словарь гиперпараметров для CatBoost
cb_params = {
    "depth": 4,
    "learning_rate": 0.06,
    "iterations": 200,  # значение по умолчанию, будет обновлено из аргументов
    "loss_function": "MultiClass",
    "custom_metric": ["Recall"],
    # Работа с категориальными признаками
    "cat_features": ["model", "car_type", "fuel_type"],
    # Регуляризация и ускорение
    "colsample_bylevel": 0.098,
    "subsample": 0.95,
    "l2_leaf_reg": 9,
    "min_data_in_leaf": 243,
    "max_bin": 187,
    "random_strength": 1,
    # Параметры ускорения
    "task_type": "CPU",  # Принудительно задаем CPU
    "thread_count": -1,
    "bootstrap_type": "Bernoulli",
    # Важное!
    "random_seed": 2025,
    "early_stopping_rounds": 50
}

# Создаём класс Config для удобного обращения к гиперпараметрам будущей модели
# можно попробовать поэкспериментировать с параметрами
@dataclass
class CFG:
    project_name: str = "ML Instruments Course"
    experiment_name: str = "Homework01_catboost"

    depth: int = 4
    learning_rate = 0.06
    loss_function = "MultiClass"
    
    # Главная фишка катбуста - работа с категориальными признаками
    custom_metric: list = None
    cat_features: list = None
    
    # Регуляризация и ускорение
    colsample_bylevel: float = 0.098
    subsample: float = 0.95
    l2_leaf_reg: int =  9
    min_data_in_leaf: int = 243
    max_bin:int= 187
    random_strength: int=1
    
    # Параметры ускорения
    task_type: str = "CPU"
    thread_count: int = -1
    bootstrap_type= "Bernoulli"
    
    # Важное!
    seed: int = 2025
    early_stopping_rounds: int= 50
        
    def __post_init__(self):
        if self.custom_metric is None:
            self.custom_metric = ["Recall"]
        if self.cat_features is None:
            self.cat_features = ["model", "car_type", "fuel_type"]

# Код проверки на присутствие необходимых переменных среды:
def check_clearml_env():
    required_env_vars = [
        "CLEARML_WEB_HOST",
        "CLEARML_API_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY"
    ]

    missing_vars = [var for var in required_env_vars if os.getenv(var) is None]
    
    if missing_vars:
        print("⚠️  Некоторые переменные среды ClearML отсутствуют.")
        for var in missing_vars:
            os.environ[var] = getpass(f"Введите значение для {var}: ")
        print("✅ Все переменные ClearML установлены.\n")

def seed_everything(seed=2025):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def dataset_prosessing():
    # Download dataset
    data = pd.read_csv('https://github.com/a-milenkin/ml_instruments/raw/refs/heads/main/data/quickstart_train.csv')

    cat_features = ["model", "car_type", "fuel_type"]  # Выделяем категориальные признаки
    targets = ["target_class", "target_reg"]
    features2drop = ["car_id"]  # эти фичи будут удалены
    
    # Отбираем итоговый набор признаков для использования моделью
    filtered_features = [col for col in data.columns if col not in targets + features2drop]
    
    # Приведение категориальных признаков к строковому типу и обработка пропусков
    for col in cat_features:
        data[col] = data[col].astype(str)
    
    num_features = [i for i in filtered_features if i not in cat_features]
    

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train = train[filtered_features]
    y_train = train["target_class"]
    
    X_test = test[filtered_features]
    y_test = test["target_class"]

    return X_train, y_train, X_test, y_test


def perform_eda(X, y, logger):
    """Проведение EDA: баланс классов и проверка на пропуски, логирование графиков в ClearML."""
    # Баланс классов
    class_counts = y.value_counts()
    plt.figure(figsize=(8, 5))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title("Распределение классов")
    plt.xlabel("Класс")
    plt.ylabel("Количество")
    plt.tight_layout()
    plt.savefig("class_balance.png")
    plt.close()
    logger.report_image("EDA", "Баланс классов", iteration=0, local_path="class_balance.png")

    
    # Проверка на пропуски
    missing_vals = X.isnull().sum()
    missing_vals_filtered = missing_vals[missing_vals > 0]
    
    if not missing_vals_filtered.empty:
        plt.figure(figsize=(10, 6))
        missing_vals_filtered.plot(kind='bar', color='salmon')
        plt.title("Пропущенные значения в признаках")
        plt.xlabel("Признак")
        plt.ylabel("Количество пропусков")
        plt.tight_layout()
        plt.savefig("missing_values.png")
        plt.close()
        logger.report_image("EDA", "Пропущенные значения", iteration=0, local_path="missing_values.png")
    else:
        print("Нет пропущенных значений в признаках.")
        logger.report_text("Нет пропущенных значений в признаках.")


def main(iterations, verbose, logger):

    # определем
    seed_everything()
    
    
    
    # Чтобы сохранить конфигурацию текущего эксперимента, перенесём её в словарь cfg_dict
    
    #  train and validation datasets
    X_train, y_train, X_test, y_test = dataset_prosessing()
    
    perform_eda(X_train, y_train, logger)

    
    # Обновляем количество итераций в параметрах CatBoost
    cb_params["iterations"] = iterations
    
    # creat model - catboost
    catboost_model = CatBoostClassifier(**cb_params)

    # Train CatBoost model
    catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=verbose, plot=False, save_snapshot=False)

    # метрики
    y_pred = catboost_model.predict(X_test)

    accuracy_score(y_test, y_pred)

    cls_report = classification_report(y_test, y_pred, target_names=y_test.unique(), output_dict=True)
    cls_report = pd.DataFrame(cls_report).T

    # save model
    catboost_model.save_model("cb_model.cbm")
    
    #классификация, логируем accuracy и F1-score
    logger.report_single_value(name='ACC', value=accuracy_score(y_test, y_pred))
    logger.report_table(title='ClassReport', series="PD with index", table_plot=cls_report)



if __name__ == "__main__":
    # проверяем переменные среды
    check_clearml_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--verbose", type=int,default=False)
    args = parser.parse_args()
    
    # Приведение аргумента verbose: если можно преобразовать в int, то делаем это, иначе False
    try:
        verbose = int(args.verbose)
    except ValueError:
        verbose = False
    
    task = Task.init(project_name="ML Instrument Course", task_name="CatBoost homework")
    logger = Logger.current_logger()
    task.add_tags(["baseline", "model", "homework"])  # Рекомендуем добавлять тэги запусков: feature_engineering, model_tuning и тд

    cfg = CFG()
    cfg_dict = asdict(cfg)
    
    # Сохраняем в нашу task'у словарь с параметрами эксперимента
    cfg_dict["iteration"] = args.iterations
    
    task.connect(cfg_dict,"Origin Config")  # Лучше всего передавать словарь,  # Комментарий, что подгружаем

    main(args.iterations, args.verbose, logger)

    # закрываем таксу clearml
    task.close()
    
