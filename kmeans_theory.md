# Теория кластеризации и алгоритм K-means

## Содержание
1. Введение в машинное обучение без учителя
2. Что такое кластеризация
3. Алгоритм K-means
4. Инициализация K-means++
5. Метрики расстояния
6. Оценка качества кластеризации
7. Бинарная и небинарная классификация
8. ROC-кривая и AUC
9. Практическая реализация

---

## 1. Введение в машинное обучение без учителя

**Машинное обучение (Machine Learning)** — область искусственного интеллекта, изучающая методы построения алгоритмов, способных обучаться.

Существует три основных типа машинного обучения:

### Обучение с учителем (Supervised Learning)
- Есть размеченные данные (входы + правильные ответы)
- Задачи: классификация, регрессия
- Примеры: распознавание цифр, предсказание цен на недвижимость

### Обучение без учителя (Unsupervised Learning)
- **Нет разметки данных**
- Алгоритм сам ищет структуру в данных
- Задачи: кластеризация, снижение размерности
- Примеры: сегментация клиентов, обнаружение аномалий

### Обучение с подкреплением (Reinforcement Learning)
- Агент учится через взаимодействие со средой
- Получает награды/штрафы за действия
- Примеры: игровые AI, роботы

**Кластеризация** относится к обучению без учителя.

---

## 2. Что такое кластеризация

### Определение
**Кластеризация** — задача разбиения множества объектов на группы (кластеры) таким образом, чтобы объекты внутри одного кластера были похожи друг на друга, а объекты из разных кластеров — различались.

### Цели кластеризации
- **Исследовательский анализ данных** — понять структуру данных
- **Сегментация** — разделить клиентов/пользователей на группы
- **Сжатие данных** — уменьшить объем данных
- **Предобработка** — подготовить данные для других алгоритмов
- **Обнаружение аномалий** — найти выбросы

### Примеры применения
1. **Маркетинг**: сегментация клиентов по поведению
2. **Биология**: группировка генов с похожей экспрессией
3. **Анализ изображений**: сжатие изображений, выделение объектов
4. **Анализ документов**: группировка похожих текстов
5. **Рекомендательные системы**: поиск похожих пользователей/товаров

### Типы кластеризации

#### Жесткая (Hard) кластеризация
Каждый объект принадлежит ровно одному кластеру.
- Пример: K-means

#### Мягкая (Soft) кластеризация
Объект может принадлежать нескольким кластерам с разными вероятностями.
- Пример: Gaussian Mixture Models (GMM)

#### Иерархическая кластеризация
Строит дерево кластеров (дендрограмму).
- Agglomerative (снизу вверх)
- Divisive (сверху вниз)

#### Плотностная кластеризация
Кластеры = области высокой плотности.
- Пример: DBSCAN

---

## 3. Алгоритм K-means

### Основная идея
K-means — простой и популярный алгоритм кластеризации, который разбивает n объектов на k кластеров, минимизируя внутрикластерную дисперсию.

### Математическая постановка
Дано:
- Множество точек X = {x₁, x₂, ..., xₙ}
- Количество кластеров k

Цель: найти k центроидов μ₁, μ₂, ..., μₖ, минимизирующих функцию:

```
J = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```

где:
- Cᵢ — i-й кластер
- μᵢ — центроид i-го кластера
- ||x - μᵢ||² — квадрат расстояния от точки до центроида

### Алгоритм (пошагово)

**Шаг 0: Инициализация**
- Случайно выбрать k начальных центроидов

**Шаг 1: Назначение кластеров (Assignment)**
- Для каждой точки xᵢ:
  - Найти ближайший центроид
  - Назначить точку этому кластеру
  
```python
for i in range(n):
    distances = [distance(x[i], centroid[j]) for j in range(k)]
    cluster[i] = argmin(distances)
```

**Шаг 2: Обновление центроидов (Update)**
- Для каждого кластера j:
  - Вычислить новый центроид как среднее всех точек кластера
  
```python
for j in range(k):
    centroid[j] = mean(points in cluster j)
```

**Шаг 3: Проверка сходимости**
- Если центроиды не изменились (или изменились незначительно):
  - **СТОП**
- Иначе:
  - Перейти к шагу 1

### Пример работы алгоритма

Дано: 6 точек на плоскости, k=2

```
Инициализация:
Точки: A(1,1), B(1,2), C(2,1), D(8,8), E(8,9), F(9,8)
Центроиды: C1(1,1), C2(9,8) — выбраны случайно

Итерация 1:
  Назначение:
    A,B,C → Кластер 1 (ближе к C1)
    D,E,F → Кластер 2 (ближе к C2)
  
  Обновление:
    C1 = mean(A,B,C) = (1.33, 1.33)
    C2 = mean(D,E,F) = (8.33, 8.33)

Итерация 2:
  Назначение: без изменений
  Центроиды не изменились → СТОП
```

### Свойства K-means

**Преимущества:**
✓ Простота реализации  
✓ Быстрая работа: O(n·k·i·d), где i — число итераций, d — размерность  
✓ Хорошо масштабируется на большие данные  
✓ Гарантированная сходимость  

**Недостатки:**
✗ Нужно заранее знать k (число кластеров)  
✗ Чувствителен к начальной инициализации  
✗ Работает только с числовыми данными  
✗ Находит только выпуклые кластеры  
✗ Чувствителен к выбросам  

### Выбор числа кластеров k

#### Метод локтя (Elbow method)
График зависимости суммы квадратов расстояний от k:
- Вычислить SSE (Sum of Squared Errors) для разных k
- Найти "локоть" — точку, где улучшение замедляется

```python
SSE = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)
plt.plot(range(1,10), SSE)
```

#### Метод силуэта (Silhouette method)
Оценивает качество кластеризации для каждого k:
- s(i) — силуэтный коэффициент для точки i
- s(i) ∈ [-1, 1]: чем ближе к 1, тем лучше

```python
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
где:
- a(i) — среднее расстояние до точек своего кластера
- b(i) — среднее расстояние до точек ближайшего другого кластера

---

## 4. Инициализация K-means++

### Проблема случайной инициализации
Стандартный K-means выбирает начальные центроиды случайно, что может привести к:
- Плохим локальным минимумам
- Медленной сходимости
- Разным результатам при разных запусках

### Алгоритм K-means++

**Идея**: выбирать начальные центроиды так, чтобы они были далеко друг от друга.

**Алгоритм:**

1. Выбрать первый центроид C₁ случайно из точек данных

2. Для каждой точки x вычислить расстояние D(x) до ближайшего уже выбранного центроида

3. Выбрать следующий центроид с вероятностью, пропорциональной D(x)²
   - Точки, далекие от центроидов, имеют больше шансов быть выбранными

4. Повторять шаги 2-3, пока не выберем k центроидов

5. Применить стандартный K-means

### Сравнение с обычным K-means

| Критерий | K-means | K-means++ |
|----------|---------|-----------|
| Инициализация | Случайная | Умная (далекие точки) |
| Сходимость | Может быть медленной | Быстрее |
| Качество | Зависит от удачи | Стабильно лучше |
| Сложность | O(1) | O(k·n) |
| Повторяемость | Разные результаты | Более стабильная |

### Псевдокод K-means++

```python
def kmeans_plus_plus(X, k):
    n = len(X)
    centroids = []
    
    # 1. Первый центроид случайно
    centroids.append(X[random.randint(0, n-1)])
    
    # 2-4. Выбрать остальные k-1 центроидов
    for _ in range(k - 1):
        # Расстояния до ближайшего центроида
        distances = []
        for x in X:
            min_dist = min([distance(x, c) for c in centroids])
            distances.append(min_dist ** 2)
        
        # Выбрать новый центроид с вероятностью ∝ D(x)²
        probabilities = distances / sum(distances)
        next_centroid = X[np.random.choice(n, p=probabilities)]
        centroids.append(next_centroid)
    
    return centroids
```

---

## 5. Метрики расстояния

Выбор метрики расстояния критичен для K-means, так как алгоритм основан на минимизации расстояний.

### 5.1 Евклидово расстояние (Euclidean Distance)

**Формула** (для 2D):
```
d(p, q) = √[(p₁-q₁)² + (p₂-q₂)²]
```

**Общий случай** (для n-мерного пространства):
```
d(p, q) = √[Σᵢ₌₁ⁿ (pᵢ - qᵢ)²]
```

**Свойства:**
- Наиболее распространенная метрика
- "Прямая" расстояние между точками
- Чувствительна к масштабу признаков
- Используется в стандартном K-means

**Пример**:
```python
p = (1, 2, 3)
q = (4, 6, 8)
d = sqrt((1-4)² + (2-6)² + (3-8)²) = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
```

**Когда использовать:**
- Данные в непрерывном пространстве
- Признаки имеют одинаковый масштаб (или нормализованы)
- Важна "геометрическая" близость

### 5.2 Манхэттенское расстояние (Manhattan Distance / L1)

**Формула**:
```
d(p, q) = Σᵢ₌₁ⁿ |pᵢ - qᵢ|
```

**Визуализация**: расстояние, которое нужно пройти по "городским кварталам" (можно двигаться только по горизонтали и вертикали).

**Пример**:
```python
p = (1, 2, 3)
q = (4, 6, 8)
d = |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
```

**Свойства:**
- Менее чувствительна к выбросам, чем евклидово
- Сумма абсолютных разностей
- Используется в k-medians кластеризации

**Когда использовать:**
- Данные в дискретном пространстве (например, координаты на сетке)
- Присутствуют выбросы
- Важна робастность

### 5.3 Расстояние Чебышева (Chebyshev Distance / L∞)

**Формула**:
```
d(p, q) = max|pᵢ - qᵢ| для всех i
```

**Визуализация**: "шахматное" расстояние — максимальное расстояние по любой координате.

**Пример**:
```python
p = (1, 2, 3)
q = (4, 6, 8)
d = max(|1-4|, |2-6|, |3-8|) = max(3, 4, 5) = 5
```

**Свойства:**
- Учитывает только самую большую разницу по координатам
- Определяет "наибольшее отклонение"
- Король в шахматах ходит на расстояние Чебышева = 1

**Когда использовать:**
- Важна максимальная разница по любому признаку
- Логистические задачи (склады, доставка)
- Игры на решетке

### 5.4 Косинусное расстояние (Cosine Distance)

**Формула**:
```
similarity(p, q) = (p · q) / (||p|| × ||q||)
distance(p, q) = 1 - similarity(p, q)
```

**Свойства:**
- Измеряет угол между векторами, а не длину
- Не зависит от масштаба векторов
- Значения от 0 (одинаковое направление) до 2 (противоположное)

**Когда использовать:**
- Текстовый анализ (сравнение документов)
- Рекомендательные системы
- Важно направление вектора, а не величина

### Сравнение метрик

| Метрика | Формула | Чувствительность к выбросам | Применение |
|---------|---------|------------------------------|------------|
| Евклидова | L2-норма | Высокая | Геометрические данные |
| Манхэттен | L1-норма | Средняя | Сеточные данные, робастность |
| Чебышев | L∞-норма | Низкая | Максимальное отклонение |
| Косинусная | Угол | Низкая | Текст, направления |

### Влияние на кластеризацию

**Визуальная форма кластеров:**
- Евклидова → круглые кластеры
- Манхэттен → ромбовидные кластеры
- Чебышев → квадратные кластеры

**Важно**: перед применением K-means рекомендуется **нормализовать** данные:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
```

---

## 6. Оценка качества кластеризации

### 6.1 Внутренние метрики (без разметки)

#### Inertia (Внутрикластерная сумма квадратов)
```
Inertia = Σᵢ₌₁ᵏ Σₓ∈Cᵢ ||x - μᵢ||²
```
- Чем меньше, тем лучше
- Всегда уменьшается с ростом k
- Используется в методе локтя

#### Silhouette Score (Силуэтный коэффициент)
```
s = (b - a) / max(a, b)
```
- Значения: [-1, 1]
- s ≈ 1: точка хорошо в своем кластере
- s ≈ 0: точка на границе кластеров
- s < 0: точка, возможно, в неправильном кластере

#### Davies-Bouldin Index
Отношение внутрикластерного расстояния к межкластерному:
- Чем меньше, тем лучше
- Учитывает как компактность, так и разделенность

#### Calinski-Harabasz Index
Отношение межкластерной дисперсии к внутрикластерной:
- Чем больше, тем лучше
- Быстро вычисляется

### 6.2 Внешние метрики (с разметкой)

Если есть истинные метки (для проверки):

#### Adjusted Rand Index (ARI)
Измеряет сходство двух разбиений:
- Значения: [-1, 1]
- ARI = 1: идеальное совпадение
- ARI = 0: случайное совпадение

#### Normalized Mutual Information (NMI)
Информационная мера сходства:
- Значения: [0, 1]
- NMI = 1: идеальное совпадение

#### Fowlkes-Mallows Index
Среднее геометрическое precision и recall:
- Значения: [0, 1]

### Пример использования метрик

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Обучение K-means
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

# Оценка качества
silhouette = silhouette_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)
calinski = calinski_harabasz_score(X, labels)

print(f"Silhouette Score: {silhouette:.3f}")  # ближе к 1 лучше
print(f"Davies-Bouldin: {davies_bouldin:.3f}")  # ближе к 0 лучше
print(f"Calinski-Harabasz: {calinski:.3f}")  # больше лучше
```

---

## 7. Бинарная и небинарная классификация

Хотя K-means — это кластеризация (неконтролируемое обучение), результаты можно использовать для классификации.

### 7.1 Бинарная классификация

**Определение**: задача разделения объектов на **два класса**.

**Примеры:**
- Спам / не спам
- Больной / здоровый
- Мошенничество / легитимная транзакция
- Положительный / отрицательный отзыв

**Особенности:**
- Только два возможных исхода
- Часто один класс — "положительный", другой — "отрицательный"
- Порог принятия решения (threshold)

**Матрица ошибок** (Confusion Matrix):

```
                   Предсказано
                 Positive | Negative
Истинное  Positive   TP   |   FN
          Negative   FP   |   TN
```

где:
- **TP** (True Positive) — правильно предсказан положительный класс
- **TN** (True Negative) — правильно предсказан отрицательный класс
- **FP** (False Positive) — ошибка 1-го рода (ложная тревога)
- **FN** (False Negative) — ошибка 2-го рода (пропуск)

**Метрики:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall (Sensitivity) = TP / (TP + FN)
Specificity = TN / (TN + FP)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**TPR и FPR** (для ROC-кривой):
```
TPR (True Positive Rate) = TP / (TP + FN) = Recall
FPR (False Positive Rate) = FP / (FP + TN) = 1 - Specificity
```

### 7.2 Небинарная (многоклассовая) классификация

**Определение**: задача разделения объектов на **три и более классов**.

**Примеры:**
- Распознавание цифр (0-9): 10 классов
- Классификация ирисов (Setosa, Versicolor, Virginica): 3 класса
- Определение жанра музыки: много классов
- Распознавание эмоций: радость, грусть, гнев и т.д.

**Особенности:**
- Более сложная задача
- Матрица ошибок размера k×k (где k — число классов)
- Метрики могут быть macro- или micro-averaged

**Подходы к многоклассовой классификации:**

1. **One-vs-Rest (OvR)**: для каждого класса строим бинарный классификатор
2. **One-vs-One (OvO)**: для каждой пары классов строим бинарный классификатор
3. **Нативные многоклассовые**: алгоритм изначально поддерживает много классов

**Матрица ошибок для 3 классов:**

```
              Предсказано
              A    B    C
Истинное  A  [20   2    1]
          B  [1   18    3]
          C  [0    2   23]
```

**Метрики для многоклассовой:**

- **Macro-averaged**: считаем метрику для каждого класса, затем усредняем
  ```
  Macro-F1 = (F1_class1 + F1_class2 + F1_class3) / 3
  ```

- **Micro-averaged**: суммируем все TP, FP, FN по всем классам
  ```
  Micro-F1 = 2 × (Micro-Precision × Micro-Recall) / (Micro-Precision + Micro-Recall)
  ```

- **Weighted-averaged**: взвешенное среднее по размеру классов

**Accuracy для многоклассовой:**
```
Accuracy = (сумма диагональных элементов) / (общее число объектов)
```

### 7.3 Связь K-means с классификацией

K-means → 2 кластера = бинарная классификация  
K-means → k кластеров = многоклассовая классификация

Однако:
- K-means не использует метки (неконтролируемое)
- Классификация использует метки (контролируемое)
- После K-means можно назначить метки кластерам и использовать для предсказания

---

## 8. ROC-кривая и AUC

### 8.1 ROC-кривая (Receiver Operating Characteristic)

**Определение**: график зависимости TPR от FPR при изменении порога классификации.

**Компоненты:**
- **Ось X**: FPR (False Positive Rate) = FP / (FP + TN)
- **Ось Y**: TPR (True Positive Rate) = TP / (TP + FN) = Recall

**Как строится:**

1. Классификатор выдает вероятности: P(class=1|x)
2. Для разных порогов t ∈ [0, 1]:
   - Если P(class=1|x) ≥ t → предсказываем класс 1
   - Иначе → предсказываем класс 0
3. Для каждого порога вычисляем TPR и FPR
4. Строим график (FPR, TPR)

**Пример:**

```
Порог | TP | FP | TN | FN | TPR  | FPR
------|----|----|----|----|------|------
 1.0  | 0  | 0  | 100| 100| 0.00 | 0.00
 0.9  | 20 | 5  | 95 | 80 | 0.20 | 0.05
 0.7  | 50 | 15 | 85 | 50 | 0.50 | 0.15
 0.5  | 70 | 30 | 70 | 30 | 0.70 | 0.30
 0.3  | 85 | 60 | 40 | 15 | 0.85 | 0.60
 0.0  | 100| 100| 0  | 0  | 1.00 | 1.00
```

График ROC-кривой:
```
TPR (Y)
  1 |        ●--------●
    |      /          
 .85|    ●            
    |   /             
 .70|  ●              
    | /               
 .50| ●               
    |/                
  0 ●------------------
    0  .15 .30 .60  1  FPR (X)
```

**Интерпретация:**

- **Идеальный классификатор**: проходит через точку (0, 1)
  - TPR = 1, FPR = 0
  - Все положительные правильно классифицированы, нет ложных срабатываний

- **Случайный классификатор**: диагональная линия (y = x)
  - Равновероятное угадывание

- **Хуже случайного**: ниже диагонали
  - Можно инвертировать предсказания и получить лучший результат

**Расстояние до идеального классификатора:**
```
distance = √[(1 - TPR)² + (0 - FPR)²]
```

### 8.2 AUC (Area Under the Curve)

**Определение**: площадь под ROC-кривой.

**Значения:**
- AUC = 1.0: идеальный классификатор
- AUC = 0.5: случайный классификатор
- AUC < 0.5: хуже случайного (нужно инвертировать)

**Интерпретация AUC:**

```
AUC = 0.9-1.0: отлично
AUC = 0.8-0.9: очень хорошо
AUC = 0.7-0.8: хорошо
AUC = 0.6-0.7: удовлетворительно
AUC = 0.5-0.6: плохо
```

**Вероятностная интерпретация:**
AUC — это вероятность того, что случайно выбранный положительный пример будет иметь более высокий score, чем случайно выбранный отрицательный.

### 8.3 Построение ROC для K-means

После применения K-means:

1. Назначить метки кластерам (если есть истинные метки)
2. Вычислить "расстояние до центроида" как score
3. Использовать расстояние как порог:
   - Близко к центроиду кластера 1 → класс 1
   - Далеко → класс 0
4. Варьировать порог и строить ROC

**Пример кода:**

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# K-means кластеризация
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(X)

# Расстояния до центроидов
distances = kmeans.transform(X)
scores = -distances[:, 1]  # Отрицательное расстояние до кластера 1

# ROC-кривая
fpr, tpr, thresholds = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)

# График
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"AUC Score: {auc:.3f}")
```

### 8.4 Выбор оптимального порога

**Метод 1: Максимизация F1-score**
```python
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
```

**Метод 2: Точка Юдена (Youden's J statistic)**
```
J = TPR - FPR
optimal_threshold = argmax(J)
```

**Метод 3: Минимизация расстояния до (0,1)**
```python
distances = np.sqrt((1-tpr)**2 + fpr**2)
optimal_idx = np.argmin(distances)
```

### 8.5 Многоклассовая ROC

Для k > 2 классов:

**Подход 1: One-vs-Rest**
- Строим k ROC-кривых
- Для каждого класса: этот класс vs все остальные

**Подход 2: One-vs-One**
- Строим k(k-1)/2 ROC-кривых
- Для каждой пары классов

**Подход 3: Micro/Macro-averaged**
```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Бинаризация меток
y_bin = label_binarize(y, classes=[0, 1, 2])

# Macro-averaged AUC
auc_macro = roc_auc_score(y_bin, y_pred_proba, average='macro')

# Micro-averaged AUC
auc_micro = roc_auc_score(y_bin, y_pred_proba, average='micro')
```

---

## 9. Практическая реализация

### 9.1 Подготовка данных

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загрузка данных
# Вариант 1: из файла
data = pd.read_csv('dataset.csv')
X = data[['feature1', 'feature2', 'feature3']].values

# Вариант 2: генерация тестовых данных
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, 
                        cluster_std=0.60, random_state=42)

# Нормализация данных (важно!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 9.2 Выбор оптимального k

```python
# Метод локтя
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', 
                    n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# График локтя
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 9.3 Обучение K-means

```python
# K-means с оптимальным k=2
kmeans = KMeans(n_clusters=2, init='k-means++', 
                n_init=10, max_iter=300, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Центроиды
centroids = kmeans.cluster_centers_

print(f"Inertia: {kmeans.inertia_:.3f}")
print(f"Number of iterations: {kmeans.n_iter_}")
```

### 9.4 Визуализация результатов

```python
plt.figure(figsize=(10, 6))

# Точки данных
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
            c=clusters, cmap='viridis', 
            marker='o', alpha=0.6, edgecolors='k')

# Центроиды
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='X', s=300, 
            edgecolors='black', linewidths=2,
            label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering Results (k=2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 9.5 Оценка качества

```python
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)

# Внутренние метрики
silhouette = silhouette_score(X_scaled, clusters)
davies_bouldin = davies_bouldin_score(X_scaled, clusters)
calinski = calinski_harabasz_score(X_scaled, clusters)

print("=== Оценка качества кластеризации ===")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {davies_bouldin:.3f}")
print(f"Calinski-Harabasz Index: {calinski:.3f}")

# Если есть истинные метки
if 'y_true' in locals():
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(y_true, clusters)
    nmi = normalized_mutual_info_score(y_true, clusters)
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Info: {nmi:.3f}")
```

### 9.6 Сравнение метрик расстояния

```python
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Функция для кластеризации с разными метриками
def cluster_with_metric(X, k, metric='euclidean'):
    if metric == 'euclidean':
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
    
    elif metric == 'manhattan':
        # K-medians (использует L1 норму)
        from sklearn_extra.cluster import KMedoids
        kmedoids = KMedoids(n_clusters=k, metric='manhattan', random_state=42)
        labels = kmedoids.fit_predict(X)
        centroids = kmedoids.cluster_centers_
    
    elif metric == 'chebyshev':
        # Ручная реализация с метрикой Чебышева
        # (упрощенная версия)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
    
    return labels, centroids

# Сравнение
metrics = ['euclidean', 'manhattan', 'chebyshev']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, metric in enumerate(metrics):
    labels, centroids = cluster_with_metric(X_scaled, k=2, metric=metric)
    
    axes[i].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                    c=labels, cmap='viridis', alpha=0.6)
    axes[i].scatter(centroids[:, 0], centroids[:, 1], 
                    c='red', marker='X', s=200)
    axes[i].set_title(f'{metric.capitalize()} Distance')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9.7 ROC-кривая для K-means

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Предполагаем бинарную классификацию (k=2)
# Используем расстояние до центроида как score
distances = kmeans.transform(X_scaled)
scores = distances[:, 0] - distances[:, 1]  # Разница расстояний

# Если есть истинные метки
if 'y_true' in locals():
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'K-means (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve for K-means Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"AUC Score: {auc:.3f}")
```

### 9.8 Матрица ошибок

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Матрица ошибок
if 'y_true' in locals():
    cm = confusion_matrix(y_true, clusters)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Cluster 0', 'Cluster 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Вычисление метрик
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("=== Метрики классификации ===")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
```

### 9.9 Полный пример для лабораторной работы

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              confusion_matrix, roc_curve, roc_auc_score)

# 1. Загрузка/генерация датасета
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                            n_redundant=0, n_clusters_per_class=1,
                            flip_y=0.1, random_state=42)

# 2. Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Выбор k методом локтя
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 4. K-means с k=2
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 5. Метрики расстояния (евклидова, манхэттен, чебышев)
from scipy.spatial.distance import euclidean, cityblock, chebyshev

distances_euclidean = []
distances_manhattan = []
distances_chebyshev = []

for i, point in enumerate(X_scaled):
    cluster_center = kmeans.cluster_centers_[clusters[i]]
    distances_euclidean.append(euclidean(point, cluster_center))
    distances_manhattan.append(cityblock(point, cluster_center))
    distances_chebyshev.append(chebyshev(point, cluster_center))

print(f"Средние расстояния:")
print(f"Евклидово: {np.mean(distances_euclidean):.3f}")
print(f"Манхэттен: {np.mean(distances_manhattan):.3f}")
print(f"Чебышева: {np.mean(distances_chebyshev):.3f}")

# 6. Бинарная классификация - матрица ошибок
cm = confusion_matrix(y, clusters)
print("\nМатрица ошибок:")
print(cm)

TP, TN = cm[1, 1], cm[0, 0]
FP, FN = cm[0, 1], cm[1, 0]

print(f"\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {(TP + TN) / (TP + TN + FP + FN):.3f}")

# 7. ROC-кривая и AUC
distances_to_cluster1 = kmeans.transform(X_scaled)[:, 1]
scores = -distances_to_cluster1

fpr, tpr, thresholds = roc_curve(y, scores)
auc = roc_auc_score(y, scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nAUC: {auc:.3f}")

# 8. Визуализация кластеров
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, 
            cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=300, linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering (k=2)')
plt.colorbar(label='Cluster')
plt.show()
```

---

## Заключение

**Основные выводы:**

1. **K-means** — простой и эффективный алгоритм кластеризации
2. **K-means++** улучшает инициализацию и качество результатов
3. **Выбор метрики расстояния** важен и зависит от данных
4. **Оценка качества** требует нескольких метрик
5. **ROC и AUC** помогают оценить качество бинарной классификации
6. **Предобработка данных** (нормализация) критична для успеха

**Дальнейшее изучение:**
- Другие алгоритмы кластеризации: DBSCAN, Hierarchical, GMM
- Уменьшение размерности: PCA, t-SNE
- Валидация результатов кластеризации
- Работа с реальными данными
