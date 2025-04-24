# 🧰 Machine Learning with ClearML and Pytorch Lightning ⚡

[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
[![pytorch](https://img.shields.io/badge/PyTorch-2.5.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

# 1. Sign Language Classification with PyTorch Lightning

Практическое применение современных инструментов Deep Learning:  
**PyTorch Lightning**, **ClearML** и других полезных утилит для трекинга, ускорения и воспроизводимости экспериментов.

---

## 📚 Оглавление

- [Описание](#описание)
- [Функциональность](#функциональность)
- [Требования](#требования)
- [Установка](#установка)
- [Как запустить](#как-запустить)
- [Структура проекта](#структура-проекта)
- [Результат](#результат)

---

## 📌 Описание

Репозиторий содержит практику по работе с нейросетями с использованием:
- ⚡ **PyTorch Lightning** — для модульной и масштабируемой тренировки моделей
- 🔭 **ClearML** — для трекинга экспериментов, логов, графиков и моделей

Цель: создать удобный, воспроизводимый и отслеживаемый pipeline для ML-задач. Реализовать классификатор языка жестов с использованием PyTorch Lightning.

Датасет содержит:

- Тренировочный набор: 27,455 образцов
- Тестовый набор: 7,172 образцов

![Датасет](data/amer_sign2.png)

---

## 🔧 Функциональность

- Загрузка и подготовка датасета (пример: Sign Language MNIST)
- Создание кастомных `LightningDataModule` и `LightningModule`
- Обучение модели с использованием `Trainer`
- Сохранение весов модели
- Прогон инференса на одном примере
- Реализация тестового прогона через `--fast_dev_run`
- Интеграция с ClearML для мониторинга (в разработке)

---

## 💻 Требования

- Python 3.8+
- PyTorch
- PyTorch Lightning
- scikit-learn
- pandas
- torchvision
- matplotlib
- requests
- clearml *(опционально)*

Установить всё можно через:

```bash
pip install -r requirements.txt
```

## Как запустить

__!Важно__

Параметр `--fast_dev_run` определён через `action="store_true"`. При таком способе использования флаг устанавливается в True , если он присутствует, и не требует передачи дополнительного значения. 
То есть, для включения режима fast_dev_run достаточно указать просто флаг без параметров.

Запуск тестового прогона:
```bash
python homework02.py --fast_dev_run
```

Запуск:
```bash
python homework02.py
```

## Структура проекта
```bash
ml_tools/
│
├── notebooks 
├── src
    ├── data/               # Данные
    ├── features            # Folder for script to build feature
    ├── models/             # Сохраненные веса моделей and sripts for train/eval
    ├── utils/              # Вспомогательные модули (при необходимости)
    ├── visualization
    └──solution             # Папка с главными файлами для запуска
        └── homework01.py   # АКТУАЛЬНЫЙ файл для запуска практики (M1_ClearML_practice-HARD)
        └── homework02.py   # АКТУАЛЬНЫЙ файл для запуска практики (M2.1_Lightning_practice)
        └── homework03.py   # АКТУАЛЬНЫЙ файл для запуска практики (M2.2_Trainer_practice)
├── logs/                   # (если используется) логи экспериментов
├── README.md               # Документация проекта
└── requirements.txt        # Зависимости проекта
```

## Результат 🎯🏆

Графики с процесса обучения^

![training_plot.png](data/training_plot.png)

Картинка с инференса инференса:

![test_picture.png](data/test_picture.png)

Обучении модели в течение 13 эпох:

- Vall acc: ~0.89

- Vall loss: ~0.1