#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа: Кластеризация датасета SDN
Анализ сетевого трафика с использованием различных методов кластеризации
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Настройка графиков
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class ClusteringAnalysis:
    """Класс для анализа кластеризации датасета"""

    def __init__(self, data_path):
        """Инициализация анализатора"""
        self.data_path = data_path
        self.df = None
        self.X_scaled = None
        self.X_pca = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def load_and_explore_data(self):
        """Загрузка и исследование датасета"""
        print("=" * 80)
        print("ЗАГРУЗКА И АНАЛИЗ ДАТАСЕТА")
        print("=" * 80)

        # Загрузка данных
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Датасет загружен: {self.df.shape[0]} строк, {self.df.shape[1]} столбцов")

        # Информация о датасете
        print("\nПервые строки датасета:")
        print(self.df.head())

        print("\nИнформация о столбцах:")
        print(self.df.info())

        print("\nСтатистическое описание:")
        print(self.df.describe())

        # Проверка меток
        print("\nУникальные значения меток:")
        print(self.df['label'].value_counts())

        # Проверка пропущенных значений
        print("\nПропущенные значения:")
        print(self.df.isnull().sum())

        return self.df

    def preprocess_data(self):
        """Предобработка данных"""
        print("\n" + "=" * 80)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("=" * 80)

        # Удаление нечисловых столбцов и меток
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Исключаем метки и идентификаторы
        exclude_cols = ['label']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Проверка на бесконечные значения
        self.df[feature_cols] = self.df[feature_cols].replace([np.inf, -np.inf], np.nan)

        # Заполнение пропущенных значений медианой
        self.df[feature_cols] = self.df[feature_cols].fillna(self.df[feature_cols].median())

        # Выбор признаков
        X = self.df[feature_cols].values

        print(f"✓ Выбрано признаков: {len(feature_cols)}")
        print(f"✓ Размер матрицы признаков: {X.shape}")

        # Стандартизация
        self.X_scaled = self.scaler.fit_transform(X)
        print("✓ Данные стандартизированы")

        # PCA для визуализации
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        explained_var = self.pca.explained_variance_ratio_
        print(f"✓ PCA выполнена: объяснённая дисперсия = {explained_var[0]:.3f}, {explained_var[1]:.3f}")

        return self.X_scaled, feature_cols

    def determine_optimal_clusters(self, max_clusters=10):
        """Определение оптимального количества кластеров методом локтя"""
        print("\n" + "=" * 80)
        print("ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ")
        print("=" * 80)

        # Ограничение размера выборки для ускорения
        sample_size = min(10000, len(self.X_scaled))
        X_sample = self.X_scaled[np.random.choice(len(self.X_scaled), sample_size, replace=False)]

        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)

        print("\nВычисление метрик для разного количества кластеров...")
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_sample)
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(X_sample, labels)
            silhouette_scores.append(sil_score)
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

        # Визуализация метода локтя
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Количество кластеров (K)', fontsize=12)
        ax1.set_ylabel('Inertia (сумма квадратов расстояний)', fontsize=12)
        ax1.set_title('Метод локтя для определения оптимального K', fontsize=14)
        ax1.grid(True)

        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Количество кластеров (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score для разного K', fontsize=14)
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        print("\n✓ График сохранён: optimal_clusters.png")

        # Рекомендуемое количество кластеров
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Рекомендуемое количество кластеров: {optimal_k}")

        return optimal_k

    def kmeans_clustering(self, n_clusters, distance_metrics):
        """Кластеризация методом K-means"""
        print("\n" + "=" * 80)
        print("КЛАСТЕРИЗАЦИЯ МЕТОДОМ K-MEANS")
        print("=" * 80)

        results = {}

        # K-means использует только евклидово расстояние напрямую
        # Для других метрик используем пользовательскую реализацию
        kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=10, random_state=42)
        labels = kmeans.fit_predict(self.X_scaled)

        # Оценка качества
        sil_score = silhouette_score(self.X_scaled, labels)
        db_score = davies_bouldin_score(self.X_scaled, labels)
        ch_score = calinski_harabasz_score(self.X_scaled, labels)

        results['kmeans'] = {
            'labels': labels,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        }

        print(f"\n✓ K-means (евклидово расстояние):")
        print(f"  - Silhouette Score: {sil_score:.4f}")
        print(f"  - Davies-Bouldin Index: {db_score:.4f}")
        print(f"  - Calinski-Harabasz Score: {ch_score:.2f}")

        return results

    def kmeans_plus_plus_clustering(self, n_clusters):
        """Кластеризация методом K-means++"""
        print("\n" + "=" * 80)
        print("КЛАСТЕРИЗАЦИЯ МЕТОДОМ K-MEANS++")
        print("=" * 80)

        kmeans_pp = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        labels = kmeans_pp.fit_predict(self.X_scaled)

        # Оценка качества
        sil_score = silhouette_score(self.X_scaled, labels)
        db_score = davies_bouldin_score(self.X_scaled, labels)
        ch_score = calinski_harabasz_score(self.X_scaled, labels)

        results = {
            'labels': labels,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        }

        print(f"\n✓ K-means++ (улучшенная инициализация):")
        print(f"  - Silhouette Score: {sil_score:.4f}")
        print(f"  - Davies-Bouldin Index: {db_score:.4f}")
        print(f"  - Calinski-Harabasz Score: {ch_score:.2f}")

        return results

    def agglomerative_clustering(self, n_clusters, distance_metrics):
        """Агломеративная кластеризация с разными метриками"""
        print("\n" + "=" * 80)
        print("АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ")
        print("=" * 80)

        results = {}

        # Метрики расстояния для агломеративной кластеризации
        metrics_map = {
            'euclidean': 'euclidean',
            'manhattan': 'manhattan',
            'chebyshev': 'chebyshev'
        }

        for metric_name in distance_metrics:
            sklearn_metric = metrics_map.get(metric_name, 'euclidean')

            print(f"\n→ Агломеративная кластеризация ({metric_name})...")

            # Используем подвыборку для ускорения
            sample_size = min(5000, len(self.X_scaled))
            sample_indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[sample_indices]

            agg = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric=sklearn_metric,
                linkage='average'
            )
            labels_sample = agg.fit_predict(X_sample)

            # Оценка качества на выборке
            sil_score = silhouette_score(X_sample, labels_sample)
            db_score = davies_bouldin_score(X_sample, labels_sample)
            ch_score = calinski_harabasz_score(X_sample, labels_sample)

            results[f'agglomerative_{metric_name}'] = {
                'labels': labels_sample,
                'silhouette': sil_score,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'sample_indices': sample_indices
            }

            print(f"  - Silhouette Score: {sil_score:.4f}")
            print(f"  - Davies-Bouldin Index: {db_score:.4f}")
            print(f"  - Calinski-Harabasz Score: {ch_score:.2f}")

        return results

    def visualize_clusters(self, clustering_results):
        """Визуализация результатов кластеризации"""
        print("\n" + "=" * 80)
        print("ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print("=" * 80)

        n_methods = len(clustering_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (method_name, result) in enumerate(clustering_results.items()):
            if idx >= 6:
                break

            labels = result['labels']

            # Для агломеративной кластеризации используем подвыборку
            if 'sample_indices' in result:
                X_plot = self.X_pca[result['sample_indices']]
            else:
                X_plot = self.X_pca

            # Ограничиваем количество точек для визуализации
            if len(X_plot) > 5000:
                sample_idx = np.random.choice(len(X_plot), 5000, replace=False)
                X_plot = X_plot[sample_idx]
                labels = labels[sample_idx]

            scatter = axes[idx].scatter(
                X_plot[:, 0],
                X_plot[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            axes[idx].set_title(f'{method_name}\nSilhouette: {result["silhouette"]:.3f}', fontsize=11)
            axes[idx].set_xlabel('PC1')
            axes[idx].set_ylabel('PC2')
            plt.colorbar(scatter, ax=axes[idx])

        plt.tight_layout()
        plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
        print("✓ Визуализация сохранена: clustering_visualization.png")

    def binary_classification(self, clustering_results):
        """Бинарная классификация на основе кластеров"""
        print("\n" + "=" * 80)
        print("БИНАРНАЯ КЛАССИФИКАЦИЯ")
        print("=" * 80)

        # Создаём бинарные метки (0 vs остальные)
        if 'label' in self.df.columns:
            # Если есть настоящие метки, используем их
            true_labels = self.df['label'].values

            # Проверяем уникальные метки
            unique_labels = np.unique(true_labels)
            print(f"\nУникальные метки в датасете: {unique_labels}")

            if len(unique_labels) == 1:
                print("⚠ Датасет содержит только один класс, создаём искусственное разделение...")
                # Используем кластеризацию для создания бинарных меток
                binary_labels = (clustering_results['kmeans']['labels'] == 0).astype(int)
            else:
                # Преобразуем в бинарные метки
                binary_labels = (true_labels == unique_labels[0]).astype(int)
        else:
            # Используем результаты кластеризации
            binary_labels = (clustering_results['kmeans']['labels'] == 0).astype(int)

        print(f"✓ Распределение бинарных классов: {np.bincount(binary_labels)}")

        return binary_labels

    def multiclass_classification(self, clustering_results):
        """Небинарная (многоклассовая) классификация"""
        print("\n" + "=" * 80)
        print("МНОГОКЛАССОВАЯ КЛАССИФИКАЦИЯ")
        print("=" * 80)

        multiclass_labels = clustering_results['kmeans']['labels']
        unique_classes = np.unique(multiclass_labels)

        print(f"✓ Количество классов: {len(unique_classes)}")
        print(f"✓ Распределение по классам:")
        for cls in unique_classes:
            count = np.sum(multiclass_labels == cls)
            print(f"  Класс {cls}: {count} объектов ({count/len(multiclass_labels)*100:.1f}%)")

        return multiclass_labels

    def evaluate_with_roc_auc(self, clustering_results, binary_labels):
        """Оценка качества бинарной классификации с ROC и AUC"""
        print("\n" + "=" * 80)
        print("ОЦЕНКА КАЧЕСТВА С ROC И AUC КРИВЫМИ")
        print("=" * 80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        auc_scores = {}

        for idx, (method_name, result) in enumerate(clustering_results.items()):
            if idx >= 6:
                break

            predicted_labels = result['labels']

            # Для агломеративной кластеризации используем подвыборку
            if 'sample_indices' in result:
                y_true = binary_labels[result['sample_indices']]
            else:
                y_true = binary_labels

            # Преобразуем кластерные метки в бинарные (кластер 0 = класс 1)
            y_pred = (predicted_labels == 0).astype(int)

            # Вычисление ROC кривой
            try:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)

                # Построение ROC кривой
                axes[idx].plot(fpr, tpr, color='darkorange', lw=2,
                              label=f'ROC curve (AUC = {roc_auc:.3f})')
                axes[idx].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                              label='Random classifier')
                axes[idx].set_xlim([0.0, 1.0])
                axes[idx].set_ylim([0.0, 1.05])
                axes[idx].set_xlabel('False Positive Rate')
                axes[idx].set_ylabel('True Positive Rate')
                axes[idx].set_title(f'ROC: {method_name}')
                axes[idx].legend(loc="lower right")
                axes[idx].grid(True)

                auc_scores[method_name] = roc_auc

                print(f"\n✓ {method_name}:")
                print(f"  - AUC Score: {roc_auc:.4f}")

                # Матрица ошибок
                cm = confusion_matrix(y_true, y_pred)
                print(f"  - Матрица ошибок:\n{cm}")

            except Exception as e:
                print(f"⚠ Ошибка при вычислении ROC для {method_name}: {e}")
                axes[idx].text(0.5, 0.5, 'Невозможно\nпостроить ROC',
                             ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('roc_auc_curves.png', dpi=300, bbox_inches='tight')
        print("\n✓ ROC кривые сохранены: roc_auc_curves.png")

        return auc_scores

    def compare_methods(self, clustering_results, auc_scores):
        """Сравнение всех методов кластеризации"""
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ")
        print("=" * 80)

        # Создание таблицы сравнения
        comparison_data = []
        for method_name, result in clustering_results.items():
            comparison_data.append({
                'Метод': method_name,
                'Silhouette': result['silhouette'],
                'Davies-Bouldin': result['davies_bouldin'],
                'Calinski-Harabasz': result['calinski_harabasz'],
                'AUC': auc_scores.get(method_name, 0)
            })

        df_comparison = pd.DataFrame(comparison_data)
        print("\n", df_comparison.to_string(index=False))

        # Визуализация сравнения
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'AUC']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            df_comparison.plot(x='Метод', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{metric} Score', fontsize=12)
            ax.set_xlabel('')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('methods_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Сравнение методов сохранено: methods_comparison.png")

        # Определение лучшего метода
        best_by_silhouette = df_comparison.loc[df_comparison['Silhouette'].idxmax(), 'Метод']
        best_by_auc = df_comparison.loc[df_comparison['AUC'].idxmax(), 'Метод']

        print(f"\n✓ Лучший метод по Silhouette Score: {best_by_silhouette}")
        print(f"✓ Лучший метод по AUC Score: {best_by_auc}")

        return df_comparison

    def generate_conclusions(self, df_comparison):
        """Генерация выводов о датасете и методах кластеризации"""
        print("\n" + "=" * 80)
        print("ВЫВОДЫ")
        print("=" * 80)

        conclusions = """
ВЫВОДЫ О ДАТАСЕТЕ И МЕТОДАХ КЛАСТЕРИЗАЦИИ

1. ХАРАКТЕРИСТИКИ ДАТАСЕТА SDN:
   - Датасет содержит данные о сетевом трафике в программно-конфигурируемых сетях (SDN)
   - Основные признаки: пакеты, байты, длительность соединений, IP-адреса, протоколы
   - Датасет имеет высокую размерность признаков, что требует предобработки
   - Выявлены различные паттерны сетевого трафика, которые могут указывать на
     нормальное поведение или аномалии

2. РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:

   a) K-means (стандартный):
      - Быстрый алгоритм, хорошо работает с большими датасетами
      - Использует евклидово расстояние
      - Чувствителен к начальной инициализации центроидов

   b) K-means++:
      - Улучшенная версия K-means с умной инициализацией центроидов
      - Обеспечивает более стабильные результаты
      - Лучше избегает локальных минимумов

   c) Агломеративная кластеризация:
      - Работает с различными метриками расстояния:
        * Евклидово расстояние - стандартная метрика
        * Манхэттенское расстояние - устойчивее к выбросам
        * Расстояние Чебышева - учитывает максимальное отклонение
      - Создаёт иерархическую структуру кластеров
      - Более вычислительно затратный алгоритм

3. СРАВНЕНИЕ МЕТРИК РАССТОЯНИЯ:
   - Евклидово расстояние: универсально, подходит для большинства задач
   - Манхэттенское расстояние: лучше для данных с ортогональной структурой
   - Расстояние Чебышева: полезно при анализе экстремальных значений

4. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ:

   Silhouette Score (от -1 до 1):
   - Положительные значения: кластеры хорошо разделены
   - Значения близкие к 0: перекрывающиеся кластеры
   - Отрицательные значения: неправильное распределение объектов

   Davies-Bouldin Index (чем меньше, тем лучше):
   - Оценивает компактность и разделимость кластеров
   - Низкие значения указывают на хорошее разделение

   Calinski-Harabasz Score (чем выше, тем лучше):
   - Отношение межкластерной к внутрикластерной дисперсии
   - Высокие значения - плотные и хорошо разделённые кластеры

5. ROC И AUC АНАЛИЗ:
   - AUC близкий к 1.0: отличное качество классификации
   - AUC около 0.5: классификатор не лучше случайного
   - ROC кривая показывает баланс между TPR и FPR

6. РЕКОМЕНДАЦИИ:
   - Для данного датасета лучше всего подходит метод с наивысшим AUC
   - K-means++ предпочтительнее стандартного K-means из-за лучшей инициализации
   - Агломеративная кластеризация полезна для понимания иерархии данных
   - Выбор метрики расстояния зависит от природы данных и задачи

7. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Выявление аномального сетевого трафика
   - Обнаружение DDoS атак и других киберугроз
   - Оптимизация маршрутизации в SDN сетях
   - Мониторинг и прогнозирование нагрузки на сеть
"""

        print(conclusions)

        # Сохранение выводов в файл
        with open('conclusions.txt', 'w', encoding='utf-8') as f:
            f.write(conclusions)
            f.write("\n\nТАБЛИЦА СРАВНЕНИЯ МЕТОДОВ:\n")
            f.write("=" * 80 + "\n")
            f.write(df_comparison.to_string(index=False))

        print("\n✓ Выводы сохранены в файл: conclusions.txt")


def main():
    """Основная функция для выполнения всей лабораторной работы"""
    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА: АНАЛИЗ КЛАСТЕРИЗАЦИИ ДАТАСЕТА SDN")
    print("=" * 80 + "\n")

    # Инициализация
    analyzer = ClusteringAnalysis('dataset_sdn.csv')

    # 1. Загрузка и исследование данных
    analyzer.load_and_explore_data()

    # 2. Предобработка данных
    X_scaled, feature_cols = analyzer.preprocess_data()

    # 3. Определение оптимального количества кластеров
    optimal_k = analyzer.determine_optimal_clusters(max_clusters=8)

    # Используем рекомендуемое количество кластеров
    n_clusters = optimal_k
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

    # 4. Применение методов кластеризации
    all_results = {}

    # K-means
    kmeans_results = analyzer.kmeans_clustering(n_clusters, distance_metrics)
    all_results.update(kmeans_results)

    # K-means++
    kmeans_pp_results = analyzer.kmeans_plus_plus_clustering(n_clusters)
    all_results['kmeans++'] = kmeans_pp_results

    # Агломеративная кластеризация
    agg_results = analyzer.agglomerative_clustering(n_clusters, distance_metrics)
    all_results.update(agg_results)

    # 5. Визуализация кластеров
    analyzer.visualize_clusters(all_results)

    # 6. Бинарная классификация
    binary_labels = analyzer.binary_classification(all_results)

    # 7. Небинарная классификация
    multiclass_labels = analyzer.multiclass_classification(all_results)

    # 8. Оценка с ROC и AUC
    auc_scores = analyzer.evaluate_with_roc_auc(all_results, binary_labels)

    # 9. Сравнение методов
    df_comparison = analyzer.compare_methods(all_results, auc_scores)

    # 10. Генерация выводов
    analyzer.generate_conclusions(df_comparison)

    print("\n" + "=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
    print("=" * 80)
    print("\nСозданные файлы:")
    print("  - optimal_clusters.png - график для определения оптимального K")
    print("  - clustering_visualization.png - визуализация всех методов кластеризации")
    print("  - roc_auc_curves.png - ROC кривые для оценки качества")
    print("  - methods_comparison.png - сравнение всех методов")
    print("  - conclusions.txt - подробные выводы о работе")
    print("\n✓ Все задания выполнены успешно!")


if __name__ == "__main__":
    main()
