#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 2: Распознавание лиц с использованием RNN
Биометрическая идентификация на основе рекуррентных нейронных сетей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Настройка для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

# Настройка графиков
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


class BiometricRNNClassifier:
    """Класс для распознавания лиц с использованием RNN/LSTM"""

    def __init__(self, img_height=50, img_width=37, min_faces_per_person=70):
        """Инициализация классификатора"""
        self.img_height = img_height
        self.img_width = img_width
        self.min_faces_per_person = min_faces_per_person
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_binarizer = LabelBinarizer()

    def load_dataset(self):
        """Загрузка датасета LFW (Labeled Faces in the Wild)"""
        print("=" * 80)
        print("ЗАГРУЗКА ДАТАСЕТА")
        print("=" * 80)

        # Загрузка датасета
        print("\n✓ Загрузка датасета LFW (Labeled Faces in the Wild)...")
        lfw_people = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=0.5,
            color=False,
            data_home='./data'
        )

        self.images = lfw_people.images
        self.labels = lfw_people.target
        self.target_names = lfw_people.target_names

        print(f"✓ Загружено изображений: {len(self.images)}")
        print(f"✓ Количество персон: {len(self.target_names)}")
        print(f"✓ Размер изображения: {self.images[0].shape}")
        print(f"✓ Распределение классов:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for i, count in enumerate(counts):
            print(f"  Персона {i} ({self.target_names[i]}): {count} изображений")

        return self.images, self.labels, self.target_names

    def normalize_data(self, X):
        """Нормализация данных"""
        print("\n" + "=" * 80)
        print("НОРМАЛИЗАЦИЯ ДАННЫХ")
        print("=" * 80)

        # Нормализация в диапазон [0, 1]
        X_normalized = X.astype('float32') / 255.0

        # Изменение формы для RNN: (samples, timesteps, features)
        # Обрабатываем изображение как последовательность строк
        n_samples = X_normalized.shape[0]
        timesteps = X_normalized.shape[1]  # Количество строк в изображении
        features = X_normalized.shape[2]   # Количество пикселей в строке

        X_reshaped = X_normalized.reshape(n_samples, timesteps, features)

        print(f"\n✓ Форма данных после нормализации: {X_reshaped.shape}")
        print(f"✓ Диапазон значений: [{X_reshaped.min():.3f}, {X_reshaped.max():.3f}]")
        print(f"✓ Интерпретация: {n_samples} изображений, {timesteps} временных шагов (строк), {features} признаков (пикселей)")

        return X_reshaped

    def split_data(self, X, y, test_size=0.3):
        """Разделение на обучающую и тестовую выборки"""
        print("\n" + "=" * 80)
        print("РАЗДЕЛЕНИЕ ДАННЫХ")
        print("=" * 80)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\n✓ Обучающая выборка: {self.X_train.shape[0]} изображений ({(1-test_size)*100:.0f}%)")
        print(f"✓ Тестовая выборка: {self.X_test.shape[0]} изображений ({test_size*100:.0f}%)")

        # One-hot encoding для многоклассовой классификации
        self.y_train_encoded = self.label_binarizer.fit_transform(self.y_train)
        self.y_test_encoded = self.label_binarizer.transform(self.y_test)

        print(f"✓ Форма меток обучающей выборки: {self.y_train_encoded.shape}")

        return self.X_train, self.X_test, self.y_train_encoded, self.y_test_encoded

    def build_rnn_model(self):
        """Построение архитектуры RNN/LSTM модели"""
        print("\n" + "=" * 80)
        print("ПОСТРОЕНИЕ АРХИТЕКТУРЫ RNN")
        print("=" * 80)

        timesteps = self.X_train.shape[1]
        features = self.X_train.shape[2]
        num_classes = len(self.target_names)

        print(f"\n✓ Параметры модели:")
        print(f"  - Временные шаги (строки изображения): {timesteps}")
        print(f"  - Признаки (пиксели в строке): {features}")
        print(f"  - Количество классов: {num_classes}")

        # Создание модели
        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=(timesteps, features)),

            # Первый LSTM слой
            layers.LSTM(128, return_sequences=True, activation='tanh'),
            layers.Dropout(0.3),

            # Второй LSTM слой
            layers.LSTM(64, return_sequences=True, activation='tanh'),
            layers.Dropout(0.3),

            # Третий LSTM слой
            layers.LSTM(32, return_sequences=False, activation='tanh'),
            layers.Dropout(0.2),

            # Полносвязные слои
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            # Выходной слой
            layers.Dense(num_classes, activation='softmax')
        ])

        # Компиляция модели
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("\n✓ Архитектура модели:")
        model.summary()

        return model

    def train_model(self, epochs=50, batch_size=32):
        """Обучение модели"""
        print("\n" + "=" * 80)
        print("ОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 80)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        print(f"\n✓ Начало обучения...")
        print(f"  - Эпох: {epochs}")
        print(f"  - Размер батча: {batch_size}")

        # Обучение
        self.history = self.model.fit(
            self.X_train, self.y_train_encoded,
            validation_data=(self.X_test, self.y_test_encoded),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("\n✓ Обучение завершено!")

        return self.history

    def plot_training_history(self):
        """Построение графиков обучения и переобучения"""
        print("\n" + "=" * 80)
        print("ВИЗУАЛИЗАЦИЯ ПРОЦЕССА ОБУЧЕНИЯ")
        print("=" * 80)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # График точности
        ax1.plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Точность модели (обучение vs валидация)')
        ax1.legend()
        ax1.grid(True)

        # График потерь
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Функция потерь (обучение vs валидация)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\n✓ График сохранён: training_history.png")

        # Анализ переобучения
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        overfitting_gap = final_train_acc - final_val_acc

        print(f"\n✓ Анализ переобучения:")
        print(f"  - Финальная точность на обучающей выборке: {final_train_acc:.4f}")
        print(f"  - Финальная точность на валидационной выборке: {final_val_acc:.4f}")
        print(f"  - Разница (признак переобучения): {overfitting_gap:.4f}")

        if overfitting_gap > 0.1:
            print("  ⚠ Обнаружено переобучение (разница > 0.1)")
        else:
            print("  ✓ Переобучение в пределах нормы")

    def calculate_metrics(self):
        """Вычисление всех метрик качества"""
        print("\n" + "=" * 80)
        print("ВЫЧИСЛЕНИЕ МЕТРИК КАЧЕСТВА")
        print("=" * 80)

        # Предсказания
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = self.y_test

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # Метрики для многоклассовой классификации
        accuracy = accuracy_score(y_true, y_pred)
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        # MSE и MAE
        mse = mean_squared_error(self.y_test_encoded, y_pred_proba)
        mae = mean_absolute_error(self.y_test_encoded, y_pred_proba)

        # Вычисление TP, TN, FP, FN для бинарного случая (One-vs-Rest)
        # Для многоклассовой задачи используем макро-усреднение
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0

        for i in range(len(self.target_names)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn

        # Усредненные значения
        n_classes = len(self.target_names)
        tp_avg = tp_total / n_classes
        tn_avg = tn_total / n_classes
        fp_avg = fp_total / n_classes
        fn_avg = fn_total / n_classes

        # TPR и FPR
        tpr = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        fpr = fp_total / (fp_total + tn_total) if (fp_total + tn_total) > 0 else 0

        # Сохранение метрик
        self.metrics = {
            'accuracy': accuracy,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mse': mse,
            'mae': mae,
            'tp': tp_avg,
            'tn': tn_avg,
            'fp': fp_avg,
            'fn': fn_avg,
            'tpr': tpr,
            'fpr': fpr
        }

        # Вывод метрик
        print("\n✓ ОСНОВНЫЕ МЕТРИКИ:")
        print(f"  - Accuracy (Точность): {accuracy:.4f}")
        print(f"  - Recall (Macro): {recall_macro:.4f}")
        print(f"  - Recall (Weighted): {recall_weighted:.4f}")
        print(f"  - F1-Score (Macro): {f1_macro:.4f}")
        print(f"  - F1-Score (Weighted): {f1_weighted:.4f}")

        print(f"\n✓ МЕТРИКИ ОШИБОК:")
        print(f"  - MSE (Mean Squared Error): {mse:.4f}")
        print(f"  - MAE (Mean Absolute Error): {mae:.4f}")

        print(f"\n✓ МАТРИЦА ОШИБОК (усредненные значения):")
        print(f"  - TP (True Positives): {tp_avg:.2f}")
        print(f"  - TN (True Negatives): {tn_avg:.2f}")
        print(f"  - FP (False Positives): {fp_avg:.2f}")
        print(f"  - FN (False Negatives): {fn_avg:.2f}")

        print(f"\n✓ ПОКАЗАТЕЛИ:")
        print(f"  - TPR (True Positive Rate): {tpr:.4f}")
        print(f"  - FPR (False Positive Rate): {fpr:.4f}")

        return self.metrics, cm

    def plot_confusion_matrix(self, cm):
        """Визуализация матрицы ошибок"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.target_names,
            yticklabels=self.target_names
        )
        plt.title('Матрица ошибок (Confusion Matrix)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Матрица ошибок сохранена: confusion_matrix.png")

    def plot_roc_curves(self):
        """Построение ROC-кривых и вычисление AUC"""
        print("\n" + "=" * 80)
        print("ROC-КРИВЫЕ И AUC")
        print("=" * 80)

        # Предсказания вероятностей
        y_pred_proba = self.model.predict(self.X_test, verbose=0)

        # Вычисление ROC-кривой для каждого класса
        n_classes = len(self.target_names)
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}

        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(
                self.y_test_encoded[:, i],
                y_pred_proba[:, i]
            )
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

        # Построение графиков
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ROC-кривые для каждого класса
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax1.plot(
                fpr_dict[i], tpr_dict[i],
                color=color, lw=2,
                label=f'{self.target_names[i]} (AUC = {roc_auc_dict[i]:.3f})'
            )

        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate (FPR)')
        ax1.set_ylabel('True Positive Rate (TPR)')
        ax1.set_title('ROC-кривые для каждого класса')
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True)

        # Сравнение TPR и FPR
        classes = [self.target_names[i] for i in range(n_classes)]
        tpr_values = [tpr_dict[i][-1] for i in range(n_classes)]
        fpr_values = [fpr_dict[i][-1] for i in range(n_classes)]

        x = np.arange(len(classes))
        width = 0.35

        ax2.bar(x - width/2, tpr_values, width, label='TPR', color='green', alpha=0.7)
        ax2.bar(x + width/2, fpr_values, width, label='FPR', color='red', alpha=0.7)
        ax2.set_xlabel('Классы')
        ax2.set_ylabel('Значение')
        ax2.set_title('Сравнение TPR и FPR по классам')
        ax2.set_xticks(x)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("\n✓ ROC-кривые сохранены: roc_curves.png")

        # Вывод AUC для каждого класса
        print("\n✓ AUC для каждого класса:")
        for i in range(n_classes):
            print(f"  - {self.target_names[i]}: {roc_auc_dict[i]:.4f}")

        # Средний AUC
        mean_auc = np.mean(list(roc_auc_dict.values()))
        print(f"\n✓ Средний AUC: {mean_auc:.4f}")

        self.metrics['auc_per_class'] = roc_auc_dict
        self.metrics['mean_auc'] = mean_auc

        return roc_auc_dict, mean_auc

    def test_on_real_examples(self, num_examples=5):
        """Тестирование на реальных примерах"""
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ПРИМЕРАХ")
        print("=" * 80)

        # Случайный выбор примеров
        indices = np.random.choice(len(self.X_test), num_examples, replace=False)

        fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))

        for idx, ax in zip(indices, axes):
            # Предсказание
            X_sample = self.X_test[idx:idx+1]
            y_pred_proba = self.model.predict(X_sample, verbose=0)
            y_pred_class = np.argmax(y_pred_proba, axis=1)[0]
            y_true_class = self.y_test[idx]

            confidence = y_pred_proba[0][y_pred_class] * 100

            # Визуализация
            # Получаем актуальные размеры из данных
            actual_height = self.X_test.shape[1]
            actual_width = self.X_test.shape[2]
            img = self.X_test[idx].reshape(actual_height, actual_width)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            true_name = self.target_names[y_true_class]
            pred_name = self.target_names[y_pred_class]

            color = 'green' if y_pred_class == y_true_class else 'red'
            ax.set_title(
                f'True: {true_name}\nPred: {pred_name}\nConf: {confidence:.1f}%',
                fontsize=9, color=color
            )

            print(f"\n✓ Пример {idx}:")
            print(f"  - Истинный класс: {true_name}")
            print(f"  - Предсказанный класс: {pred_name}")
            print(f"  - Уверенность: {confidence:.2f}%")
            print(f"  - Результат: {'✓ Правильно' if y_pred_class == y_true_class else '✗ Ошибка'}")

        plt.tight_layout()
        plt.savefig('test_examples.png', dpi=300, bbox_inches='tight')
        print("\n✓ Примеры сохранены: test_examples.png")

    def generate_report(self):
        """Генерация отчета с выводами"""
        print("\n" + "=" * 80)
        print("ГЕНЕРАЦИЯ ОТЧЕТА")
        print("=" * 80)

        report = f"""
ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ 2: RNN ДЛЯ БИОМЕТРИЧЕСКОЙ ИДЕНТИФИКАЦИИ
{"=" * 80}

1. ОПИСАНИЕ ДАТАСЕТА
   - Датасет: LFW (Labeled Faces in the Wild)
   - Количество изображений: {len(self.images)}
   - Количество классов (персон): {len(self.target_names)}
   - Размер изображения: {self.img_height}x{self.img_width} пикселей
   - Тип биометрии: Статическая (изображения лиц)

2. АРХИТЕКТУРА RNN
   - Тип: LSTM (Long Short-Term Memory)
   - Количество LSTM слоев: 3
   - Размеры LSTM: 128 -> 64 -> 32 нейронов
   - Dropout: 0.2-0.3 для регуляризации
   - Активация: tanh (LSTM), relu (Dense), softmax (выход)
   - Входные данные: Изображение обрабатывается как последовательность строк пикселей

3. ОБУЧЕНИЕ
   - Процент обучающей выборки: {(1 - 0.3) * 100:.0f}%
   - Процент тестовой выборки: {0.3 * 100:.0f}%
   - Оптимизатор: Adam
   - Функция потерь: Categorical Crossentropy
   - Early Stopping: Да (patience=10)
   - Learning Rate Reduction: Да (factor=0.5, patience=5)

4. РЕЗУЛЬТАТЫ ОБУЧЕНИЯ
   - Финальная точность на обучающей выборке: {self.history.history['accuracy'][-1]:.4f}
   - Финальная точность на валидационной выборке: {self.history.history['val_accuracy'][-1]:.4f}
   - Признаки переобучения: {self.history.history['accuracy'][-1] - self.history.history['val_accuracy'][-1]:.4f}

5. МЕТРИКИ КАЧЕСТВА
   - Accuracy: {self.metrics['accuracy']:.4f}
   - Recall (Macro): {self.metrics['recall_macro']:.4f}
   - F1-Score (Macro): {self.metrics['f1_macro']:.4f}
   - TPR (True Positive Rate): {self.metrics['tpr']:.4f}
   - FPR (False Positive Rate): {self.metrics['fpr']:.4f}
   - MSE (Mean Squared Error): {self.metrics['mse']:.4f}
   - MAE (Mean Absolute Error): {self.metrics['mae']:.4f}
   - Mean AUC: {self.metrics['mean_auc']:.4f}

6. МАТРИЦА ОШИБОК (усредненные значения)
   - TP (True Positives): {self.metrics['tp']:.2f}
   - TN (True Negatives): {self.metrics['tn']:.2f}
   - FP (False Positives): {self.metrics['fp']:.2f}
   - FN (False Negatives): {self.metrics['fn']:.2f}

7. ВЫВОДЫ

   7.1. Эффективность RNN для биометрии:
        - LSTM успешно справляется с задачей распознавания лиц
        - Обработка изображения как последовательности строк позволяет учитывать
          пространственные зависимости между пикселями
        - Рекуррентная архитектура эффективна для выделения паттернов в данных

   7.2. Качество модели:
        - Accuracy {self.metrics['accuracy']:.4f} указывает на {'хорошее' if self.metrics['accuracy'] > 0.8 else 'удовлетворительное'} качество классификации
        - F1-Score {self.metrics['f1_macro']:.4f} показывает {'сбалансированную' if self.metrics['f1_macro'] > 0.75 else 'несбалансированную'} работу модели
        - AUC {self.metrics['mean_auc']:.4f} свидетельствует о {'хорошей' if self.metrics['mean_auc'] > 0.85 else 'удовлетворительной'} разделимости классов

   7.3. Переобучение:
        - Разница между train и validation accuracy составляет {self.history.history['accuracy'][-1] - self.history.history['val_accuracy'][-1]:.4f}
        - {'Переобучение минимально благодаря Dropout и Early Stopping' if abs(self.history.history['accuracy'][-1] - self.history.history['val_accuracy'][-1]) < 0.1 else 'Наблюдается умеренное переобучение, требуется дополнительная регуляризация'}

   7.4. TPR vs FPR:
        - TPR = {self.metrics['tpr']:.4f} - модель корректно распознает {'большинство' if self.metrics['tpr'] > 0.8 else 'значительную часть'} положительных примеров
        - FPR = {self.metrics['fpr']:.4f} - {'низкий' if self.metrics['fpr'] < 0.2 else 'умеренный'} уровень ложных срабатываний
        - Соотношение TPR/FPR = {self.metrics['tpr']/self.metrics['fpr'] if self.metrics['fpr'] > 0 else 'inf':.2f} - {'отличный' if self.metrics['tpr']/self.metrics['fpr'] > 5 else 'хороший'} баланс

   7.5. Практическое применение:
        - Модель может использоваться для систем контроля доступа
        - Подходит для верификации личности в биометрических системах
        - Требуется дополнительная настройка для промышленного использования

   7.6. Рекомендации по улучшению:
        - Увеличение датасета для лучшей генерализации
        - Использование аугментации данных
        - Экспериментирование с гибридными CNN+RNN архитектурами
        - Применение attention механизмов
        - Тонкая настройка гиперпараметров

8. ЗАКЛЮЧЕНИЕ
   RNN/LSTM архитектура продемонстрировала {'высокую' if self.metrics['accuracy'] > 0.85 else 'хорошую'} эффективность
   в задаче биометрической идентификации по изображениям лиц. Модель успешно
   обучена и может быть использована для практических применений с учетом
   рекомендаций по улучшению.

{"=" * 80}
"""

        # Сохранение отчета
        with open('report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\n✓ Отчет сохранен: report.txt")
        print(report)


def main():
    """Основная функция для выполнения лабораторной работы"""
    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА 2: RNN ДЛЯ БИОМЕТРИЧЕСКОЙ ИДЕНТИФИКАЦИИ")
    print("=" * 80)

    # Инициализация классификатора
    classifier = BiometricRNNClassifier(
        img_height=50,
        img_width=37,
        min_faces_per_person=70
    )

    try:
        # 1. Загрузка датасета
        images, labels, target_names = classifier.load_dataset()

        # 2. Нормализация данных
        X_normalized = classifier.normalize_data(images)

        # 3. Разделение на train/test
        X_train, X_test, y_train, y_test = classifier.split_data(
            X_normalized, labels, test_size=0.3
        )

        # 4. Построение архитектуры RNN
        model = classifier.build_rnn_model()

        # 5. Обучение модели
        history = classifier.train_model(epochs=50, batch_size=32)

        # 6. Визуализация обучения и анализ переобучения
        classifier.plot_training_history()

        # 7. Вычисление метрик
        metrics, cm = classifier.calculate_metrics()

        # 8. Визуализация матрицы ошибок
        classifier.plot_confusion_matrix(cm)

        # 9. ROC-кривые и AUC
        roc_auc, mean_auc = classifier.plot_roc_curves()

        # 10. Тестирование на реальных примерах
        classifier.test_on_real_examples(num_examples=5)

        # 11. Генерация отчета
        classifier.generate_report()

        print("\n" + "=" * 80)
        print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 80)
        print("\n✓ Созданные файлы:")
        print("  - training_history.png - график обучения и переобучения")
        print("  - confusion_matrix.png - матрица ошибок")
        print("  - roc_curves.png - ROC-кривые и сравнение TPR/FPR")
        print("  - test_examples.png - примеры тестирования")
        print("  - report.txt - полный отчет с выводами")

    except Exception as e:
        print(f"\n✗ Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
