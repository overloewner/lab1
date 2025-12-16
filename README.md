# Лабораторная работа 3: CNN для биометрической идентификации

## Описание

Реализация сверточной нейронной сети (CNN) для биометрической идентификации личности по фотографиям лиц.

## Датасет

**LFW (Labeled Faces in the Wild)** - биометрический датасет с фотографиями лиц известных личностей:
- Размер: 1288 изображений
- Количество классов: 7 персон
- Размер изображения: 62x47 пикселей (grayscale)

## Архитектура CNN

```
Input (62x47x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Dense(128) → BatchNorm → Dropout(0.5)
    ↓
Dense(7, softmax)
```

**Параметры:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau

## Метрики

Скрипт вычисляет полный набор метрик:
- **Accuracy**: Общая точность классификации
- **Recall**: Полнота (macro)
- **F1-Score**: F-мера (macro)
- **TPR**: True Positive Rate
- **FPR**: False Positive Rate
- **TP/TN/FP/FN**: Confusion matrix элементы (усредненные)
- **MSE/MAE**: Ошибки предсказания вероятностей
- **AUC**: Area Under ROC Curve

## Визуализации

1. **training_history.png** - График обучения (accuracy/loss) с анализом переобучения
2. **confusion_matrix.png** - Матрица ошибок
3. **roc_curves.png** - ROC-кривые для всех классов
4. **test_examples.png** - Тестирование на реальных примерах

## Использование

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
python3 cnn_biometric_recognition.py
```

## Результаты

Модель обеспечивает:
- Эффективное распознавание лиц
- Минимальное переобучение
- Высокое качество классификации

Детальный отчет генерируется в файле `report.txt`.

## Преимущества CNN

- Автоматическое извлечение признаков
- Инвариантность к сдвигам и деформациям
- Эффективное обучение с регуляризацией
- Применимость в реальных системах безопасности
