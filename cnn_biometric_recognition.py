#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа 3: Биометрическая идентификация на основе CNN
Сверточная нейронная сеть для распознавания лиц
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Установка seed для воспроизводимости
np.random.seed(42)
tf.random.set_seed(42)

class CNNBiometricRecognition:
    """
    Класс для биометрической идентификации на основе CNN
    """

    def __init__(self):
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        self.class_names = None

    def load_dataset(self):
        """
        Загрузка датасета LFW (Labeled Faces in the Wild)
        Биометрический датасет с фотографиями лиц известных личностей
        """
        print("=" * 70)
        print("ЗАГРУЗКА БИОМЕТРИЧЕСКОГО ДАТАСЕТА")
        print("=" * 70)

        # Загрузка датасета LFW
        lfw_people = fetch_lfw_people(
            min_faces_per_person=70,
            resize=0.5,
            color=False,
            data_home='./data'
        )

        # Получение данных
        self.images = lfw_people.images
        self.labels = lfw_people.target
        self.class_names = lfw_people.target_names

        self.img_height = lfw_people.images.shape[1]
        self.img_width = lfw_people.images.shape[2]

        print(f"Датасет: LFW (Labeled Faces in the Wild)")
        print(f"Количество изображений: {len(self.images)}")
        print(f"Размер изображения: {self.img_height}x{self.img_width}")
        print(f"Количество классов (персон): {len(self.class_names)}")
        print(f"Классы: {', '.join(self.class_names)}")

        # Подсчет количества изображений по классам
        unique, counts = np.unique(self.labels, return_counts=True)
        print("\nРаспределение по классам:")
        for name, count in zip(self.class_names, counts):
            print(f"  {name}: {count} изображений")

        return self.images, self.labels

    def prepare_data(self, test_size=0.25):
        """
        Подготовка данных для CNN
        Reshape для Conv2D: (samples, height, width, channels)
        """
        print("\n" + "=" * 70)
        print("ПОДГОТОВКА ДАННЫХ ДЛЯ CNN")
        print("=" * 70)

        # Нормализация данных [0, 1]
        X = self.images.astype('float32') / 255.0

        # Добавление канала для grayscale изображений
        # Из (samples, height, width) -> (samples, height, width, 1)
        X = X.reshape(-1, self.img_height, self.img_width, 1)

        # Кодирование меток
        y = self.label_encoder.fit_transform(self.labels)

        # Разделение на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Размер обучающей выборки: {self.X_train.shape[0]}")
        print(f"Размер тестовой выборки: {self.X_test.shape[0]}")
        print(f"Форма входных данных: {self.X_train.shape[1:]}")
        print(f"Количество каналов: {self.X_train.shape[-1]} (grayscale)")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_cnn_model(self):
        """
        Построение сверточной нейронной сети (CNN)
        Архитектура:
        - 3 блока Conv2D + MaxPooling2D + Dropout
        - Flatten
        - Dense слои с Dropout
        - Softmax для multi-class классификации
        """
        print("\n" + "=" * 70)
        print("ПОСТРОЕНИЕ CNN АРХИТЕКТУРЫ")
        print("=" * 70)

        num_classes = len(np.unique(self.y_train))
        input_shape = (self.img_height, self.img_width, 1)

        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=input_shape),

            # Первый сверточный блок
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Второй сверточный блок
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Третий сверточный блок
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Flatten и полносвязные слои
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Выходной слой
            layers.Dense(num_classes, activation='softmax')
        ])

        # Компиляция модели
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model

        print("\nАрхитектура сверточной нейронной сети:")
        print("-" * 70)
        self.model.summary()

        # Подсчет параметров
        total_params = self.model.count_params()
        print(f"\nВсего параметров: {total_params:,}")

        return self.model

    def train_model(self, epochs=50, batch_size=32):
        """
        Обучение CNN модели с анализом переобучения
        """
        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ CNN МОДЕЛИ")
        print("=" * 70)

        # Callbacks для предотвращения переобучения
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print(f"Эпох: {epochs}")
        print(f"Размер батча: {batch_size}")
        print("Callbacks: EarlyStopping, ReduceLROnPlateau")
        print("\nНачало обучения...")
        print("-" * 70)

        # Обучение
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        print("\n" + "=" * 70)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 70)

        return self.history

    def evaluate_model(self):
        """
        Полная оценка модели со всеми метриками
        """
        print("\n" + "=" * 70)
        print("ОЦЕНКА МОДЕЛИ")
        print("=" * 70)

        # Предсказания
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Основные метрики
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred, average='macro')
        f1 = f1_score(self.y_test, y_pred, average='macro')

        # Confusion Matrix для TPR, FPR, TP, TN, FP, FN
        cm = confusion_matrix(self.y_test, y_pred)

        # Вычисление TP, TN, FP, FN для multi-class (усредненные)
        TP = np.diag(cm).sum() / len(np.unique(self.y_test))
        TN = (cm.sum() - cm.sum(axis=1) - cm.sum(axis=0) + np.diag(cm)).sum() / len(np.unique(self.y_test))
        FP = (cm.sum(axis=0) - np.diag(cm)).sum() / len(np.unique(self.y_test))
        FN = (cm.sum(axis=1) - np.diag(cm)).sum() / len(np.unique(self.y_test))

        # TPR (True Positive Rate) = Recall
        TPR = recall

        # FPR (False Positive Rate)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        # MSE и MAE
        # Для multi-class используем one-hot encoded labels
        y_test_onehot = keras.utils.to_categorical(self.y_test, num_classes=len(np.unique(self.y_test)))
        mse = mean_squared_error(y_test_onehot, y_pred_proba)
        mae = mean_absolute_error(y_test_onehot, y_pred_proba)

        # ROC AUC для каждого класса
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))

        auc_scores = []
        for i in range(y_test_bin.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)

        mean_auc = np.mean(auc_scores)

        print("\nМЕТРИКИ КЛАССИФИКАЦИИ:")
        print("-" * 70)
        print(f"Accuracy (Точность):              {accuracy:.4f}")
        print(f"Recall (Полнота, macro):          {recall:.4f}")
        print(f"F1-Score (macro):                 {f1:.4f}")
        print(f"TPR (True Positive Rate):         {TPR:.4f}")
        print(f"FPR (False Positive Rate):        {FPR:.4f}")
        print(f"TP (True Positives, avg):         {TP:.2f}")
        print(f"TN (True Negatives, avg):         {TN:.2f}")
        print(f"FP (False Positives, avg):        {FP:.2f}")
        print(f"FN (False Negatives, avg):        {FN:.2f}")
        print(f"MSE (Mean Squared Error):         {mse:.4f}")
        print(f"MAE (Mean Absolute Error):        {mae:.4f}")
        print(f"AUC (Area Under ROC Curve):       {mean_auc:.4f}")

        # Анализ переобучения
        train_acc = self.history.history['accuracy'][-1]
        val_acc = self.history.history['val_accuracy'][-1]
        overfitting = train_acc - val_acc

        print("\nАНАЛИЗ ПЕРЕОБУЧЕНИЯ:")
        print("-" * 70)
        print(f"Train Accuracy:                   {train_acc:.4f}")
        print(f"Validation Accuracy:              {val_acc:.4f}")
        print(f"Разница (Overfitting):            {overfitting:.4f}")

        if overfitting < 0.05:
            print("Оценка: Переобучение минимальное ✓")
        elif overfitting < 0.15:
            print("Оценка: Умеренное переобучение")
        else:
            print("Оценка: Сильное переобучение ✗")

        # Сохранение метрик
        self.metrics = {
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'TPR': TPR,
            'FPR': FPR,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'MSE': mse,
            'MAE': mae,
            'AUC': mean_auc,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfitting': overfitting,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        return self.metrics

    def plot_training_history(self):
        """
        Визуализация процесса обучения с анализом переобучения
        """
        print("\n" + "=" * 70)
        print("ВИЗУАЛИЗАЦИЯ: История обучения")
        print("=" * 70)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy (Overfitting Analysis)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Loss (Overfitting Analysis)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Сохранено: training_history.png")
        plt.close()

    def plot_confusion_matrix(self):
        """
        Визуализация матрицы ошибок
        """
        print("\n" + "=" * 70)
        print("ВИЗУАЛИЗАЦИЯ: Confusion Matrix")
        print("=" * 70)

        cm = self.metrics['confusion_matrix']

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - CNN Biometric Recognition', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Сохранено: confusion_matrix.png")
        plt.close()

    def plot_roc_curves(self):
        """
        Визуализация ROC-кривых для всех классов
        """
        print("\n" + "=" * 70)
        print("ВИЗУАЛИЗАЦИЯ: ROC Curves")
        print("=" * 70)

        from sklearn.preprocessing import label_binarize

        y_test_bin = label_binarize(self.y_test, classes=np.unique(self.y_test))
        y_pred_proba = self.metrics['y_pred_proba']

        n_classes = y_test_bin.shape[1]

        plt.figure(figsize=(12, 8))

        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title('ROC Curves - CNN Biometric Recognition', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Сохранено: roc_curves.png")
        plt.close()

    def test_on_real_examples(self, num_examples=5):
        """
        Тестирование на реальных примерах из тестовой выборки
        """
        print("\n" + "=" * 70)
        print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ПРИМЕРАХ")
        print("=" * 70)

        # Выбор случайных примеров
        indices = np.random.choice(len(self.X_test), num_examples, replace=False)

        fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))

        for i, idx in enumerate(indices):
            # Предсказание
            img_input = self.X_test[idx:idx+1]
            pred_proba = self.model.predict(img_input, verbose=0)
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[0][pred_class]

            true_class = self.y_test[idx]

            # Получение изображения для отображения
            img = self.X_test[idx].reshape(self.img_height, self.img_width)

            # Визуализация
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

            pred_name = self.class_names[pred_class]
            true_name = self.class_names[true_class]

            color = 'green' if pred_class == true_class else 'red'

            title = f"True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2%}"
            axes[i].set_title(title, fontsize=9, color=color, fontweight='bold')

            # Вывод в консоль
            status = "✓ CORRECT" if pred_class == true_class else "✗ WRONG"
            print(f"\nПример {i+1}: {status}")
            print(f"  Истинный класс:  {true_name}")
            print(f"  Предсказанный:   {pred_name}")
            print(f"  Уверенность:     {confidence:.2%}")

        plt.tight_layout()
        plt.savefig('test_examples.png', dpi=300, bbox_inches='tight')
        print("\n✓ Сохранено: test_examples.png")
        plt.close()

    def generate_report(self):
        """
        Генерация текстового отчета
        """
        print("\n" + "=" * 70)
        print("ГЕНЕРАЦИЯ ОТЧЕТА")
        print("=" * 70)

        report = []
        report.append("=" * 70)
        report.append("ОТЧЕТ: БИОМЕТРИЧЕСКАЯ ИДЕНТИФИКАЦИЯ НА ОСНОВЕ CNN")
        report.append("Лабораторная работа 3")
        report.append("=" * 70)
        report.append("")

        report.append("1. ДАТАСЕТ")
        report.append("-" * 70)
        report.append(f"Название: LFW (Labeled Faces in the Wild)")
        report.append(f"Тип: Биометрические данные (фотографии лиц)")
        report.append(f"Количество изображений: {len(self.images)}")
        report.append(f"Размер изображения: {self.img_height}x{self.img_width} пикселей")
        report.append(f"Количество классов: {len(self.class_names)}")
        report.append(f"Классы: {', '.join(self.class_names)}")
        report.append("")

        report.append("2. АРХИТЕКТУРА CNN")
        report.append("-" * 70)
        report.append("Тип: Сверточная нейронная сеть (Convolutional Neural Network)")
        report.append("Структура:")
        report.append("  - Входной слой: {}x{}x1 (grayscale)".format(self.img_height, self.img_width))
        report.append("  - Блок 1: Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)")
        report.append("  - Блок 2: Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)")
        report.append("  - Блок 3: Conv2D(128) → BatchNorm → Conv2D(128) → MaxPool → Dropout(0.25)")
        report.append("  - Flatten")
        report.append("  - Dense(256) → BatchNorm → Dropout(0.5)")
        report.append("  - Dense(128) → BatchNorm → Dropout(0.5)")
        report.append("  - Выходной слой: Dense({}) + Softmax".format(len(self.class_names)))
        report.append(f"Всего параметров: {self.model.count_params():,}")
        report.append("")

        report.append("3. ОБУЧЕНИЕ")
        report.append("-" * 70)
        report.append(f"Размер обучающей выборки: {len(self.X_train)}")
        report.append(f"Размер тестовой выборки: {len(self.X_test)}")
        report.append(f"Optimizer: Adam (lr=0.001)")
        report.append(f"Loss function: Sparse Categorical Crossentropy")
        report.append(f"Callbacks: EarlyStopping, ReduceLROnPlateau")
        report.append("")

        report.append("4. МЕТРИКИ КАЧЕСТВА")
        report.append("-" * 70)
        report.append(f"Accuracy:              {self.metrics['accuracy']:.4f}")
        report.append(f"Recall (macro):        {self.metrics['recall']:.4f}")
        report.append(f"F1-Score (macro):      {self.metrics['f1']:.4f}")
        report.append(f"TPR:                   {self.metrics['TPR']:.4f}")
        report.append(f"FPR:                   {self.metrics['FPR']:.4f}")
        report.append(f"TP (avg):              {self.metrics['TP']:.2f}")
        report.append(f"TN (avg):              {self.metrics['TN']:.2f}")
        report.append(f"FP (avg):              {self.metrics['FP']:.2f}")
        report.append(f"FN (avg):              {self.metrics['FN']:.2f}")
        report.append(f"MSE:                   {self.metrics['MSE']:.4f}")
        report.append(f"MAE:                   {self.metrics['MAE']:.4f}")
        report.append(f"AUC:                   {self.metrics['AUC']:.4f}")
        report.append("")

        report.append("5. АНАЛИЗ ПЕРЕОБУЧЕНИЯ")
        report.append("-" * 70)
        report.append(f"Train Accuracy:        {self.metrics['train_acc']:.4f}")
        report.append(f"Validation Accuracy:   {self.metrics['val_acc']:.4f}")
        report.append(f"Разница (Overfitting): {self.metrics['overfitting']:.4f}")

        if self.metrics['overfitting'] < 0.05:
            report.append("Оценка: Переобучение минимальное ✓")
        elif self.metrics['overfitting'] < 0.15:
            report.append("Оценка: Умеренное переобучение")
        else:
            report.append("Оценка: Сильное переобучение ✗")
        report.append("")

        report.append("6. ВЫВОДЫ")
        report.append("-" * 70)
        report.append("Разработана сверточная нейронная сеть для биометрической идентификации")
        report.append("по фотографиям лиц из датасета LFW.")
        report.append("")
        report.append("Основные достижения:")
        report.append(f"  • Точность классификации: {self.metrics['accuracy']:.1%}")
        report.append(f"  • AUC-ROC: {self.metrics['AUC']:.4f}")
        report.append(f"  • Минимальное переобучение: {self.metrics['overfitting']:.4f}")
        report.append("")
        report.append("CNN показывает эффективность в задачах распознавания лиц благодаря:")
        report.append("  • Автоматическому извлечению признаков через сверточные слои")
        report.append("  • Инвариантности к небольшим сдвигам и деформациям")
        report.append("  • BatchNormalization для стабилизации обучения")
        report.append("  • Dropout для предотвращения переобучения")
        report.append("")
        report.append("Практическое применение:")
        report.append("  • Системы контроля доступа")
        report.append("  • Автоматическая маркировка фото")
        report.append("  • Поиск людей по фотографиям")
        report.append("")
        report.append("=" * 70)

        # Сохранение отчета
        report_text = "\n".join(report)
        with open('report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\n✓ Отчет сохранен: report.txt")

        return report_text


def main():
    """
    Основная функция запуска
    """
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  ЛАБОРАТОРНАЯ РАБОТА 3: CNN ДЛЯ БИОМЕТРИЧЕСКОЙ ИДЕНТИФИКАЦИИ  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")

    # Создание экземпляра класса
    cnn = CNNBiometricRecognition()

    # 1. Загрузка датасета
    cnn.load_dataset()

    # 2. Подготовка данных
    cnn.prepare_data(test_size=0.25)

    # 3. Построение модели
    cnn.build_cnn_model()

    # 4. Обучение
    cnn.train_model(epochs=50, batch_size=32)

    # 5. Оценка
    cnn.evaluate_model()

    # 6. Визуализации
    cnn.plot_training_history()
    cnn.plot_confusion_matrix()
    cnn.plot_roc_curves()
    cnn.test_on_real_examples(num_examples=5)

    # 7. Генерация отчета
    cnn.generate_report()

    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  РАБОТА ЗАВЕРШЕНА УСПЕШНО  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")


if __name__ == "__main__":
    main()
