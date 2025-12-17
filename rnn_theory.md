# Теория рекуррентных нейронных сетей (RNN) для биометрии

## Содержание
1. Введение в нейронные сети
2. От обычных сетей к рекуррентным
3. Архитектура RNN
4. Типы RNN (LSTM, GRU)
5. Обучение RNN
6. Биометрия и RNN
7. Разделение данных и переобучение
8. Метрики оценки
9. Практическая реализация

---

## 1. Введение в нейронные сети

### 1.1 Что такое нейронная сеть?

**Нейронная сеть** — математическая модель, вдохновленная работой биологических нейронов, способная обучаться распознавать сложные паттерны в данных.

### 1.2 Искусственный нейрон

**Математическая модель:**

```
y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
  = f(Σᵢ wᵢxᵢ + b)
  = f(w·x + b)
```

где:
- **x** — вход (входные признаки)
- **w** — веса (параметры, которые учатся)
- **b** — смещение (bias)
- **f** — функция активации
- **y** — выход

**Функции активации:**

1. **Sigmoid**: σ(x) = 1 / (1 + e⁻ˣ)
   - Выход: (0, 1)
   - Проблема: затухание градиентов

2. **Tanh**: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
   - Выход: (-1, 1)
   - Центрирован относительно 0

3. **ReLU**: f(x) = max(0, x)
   - Выход: [0, +∞)
   - Самая популярная для скрытых слоев

4. **Softmax**: σ(xᵢ) = eˣⁱ / Σⱼ eˣʲ
   - Выход: вероятности (сумма = 1)
   - Используется в последнем слое для классификации

### 1.3 Многослойная сеть (MLP)

**Архитектура:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Прямое распространение (Forward Propagation):**

1. **Входной слой**: принимает данные
2. **Скрытые слои**: 
   ```
   h⁽¹⁾ = f(W⁽¹⁾x + b⁽¹⁾)
   h⁽²⁾ = f(W⁽²⁾h⁽¹⁾ + b⁽²⁾)
   ```
3. **Выходной слой**: предсказание
   ```
   y = f(W⁽ᴸ⁾h⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾)
   ```

**Обучение:**
- **Функция потерь**: измеряет ошибку
  - MSE (регрессия): L = (y - ŷ)²
  - Cross-Entropy (классификация): L = -Σ yᵢlog(ŷᵢ)

- **Обратное распространение (Backpropagation)**: вычисляет градиенты
- **Градиентный спуск**: обновляет веса
  ```
  w := w - η∇L(w)
  ```
  где η — learning rate (скорость обучения)

---

## 2. От обычных сетей к рекуррентным

### 2.1 Ограничения MLP

**Проблема**: MLP обрабатывает фиксированный вход и не имеет памяти.

**Не подходит для:**
- Текстов переменной длины
- Временных рядов
- Видео (последовательность кадров)
- Речи (последовательность звуков)

**Пример:**
```
Задача: перевести "I love AI" → "Я люблю ИИ"
MLP: обрабатывает все слова одновременно, теряя порядок
```

### 2.2 Последовательные данные

**Определение**: данные, где важен **порядок** элементов.

**Примеры:**
- Текст: последовательность слов
- Речь: последовательность звуковых сигналов
- Видео: последовательность кадров
- Временные ряды: последовательность значений
- Поведенческая биометрия: последовательность действий

**Свойства:**
- Элементы связаны друг с другом
- Изменение порядка меняет смысл
- Длина может быть переменной

### 2.3 Идея рекуррентности

**Рекуррентность** = обратная связь = память о прошлом

**Ключевая идея RNN:**
```
Выход в момент t зависит от:
1. Входа в момент t
2. Состояния в момент t-1 (память о прошлом)
```

**Визуально:**
```
      ↓ input
    [RNN] ← (обратная связь)
      ↓ output
```

---

## 3. Архитектура RNN

### 3.1 Базовая структура

**Развернутая во времени:**
```
x₀ → [h₀] → [h₁] → [h₂] → [h₃]
       ↓      ↓      ↓      ↓
      y₀     y₁     y₂     y₃
```

**Формулы:**
```
hₜ = f(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
yₜ = g(Wₕᵧ·hₜ + bᵧ)
```

где:
- **xₜ** — вход в момент t
- **hₜ** — скрытое состояние (hidden state) в момент t
- **yₜ** — выход в момент t
- **Wₓₕ** — веса вход → скрытое состояние
- **Wₕₕ** — веса скрытое состояние → скрытое состояние (рекуррентные)
- **Wₕᵧ** — веса скрытое состояние → выход
- **f, g** — функции активации (обычно tanh, sigmoid, ReLU)

**Важно**: веса **W** и **b** одинаковые для всех временных шагов!

### 3.2 Типы архитектур RNN

#### One-to-One (не рекуррентная)
```
Input → [NN] → Output
```
Обычная нейронная сеть

#### One-to-Many
```
Input → [RNN] → [RNN] → [RNN]
           ↓       ↓       ↓
         Out1    Out2    Out3
```
**Пример**: генерация изображения по описанию, музыкальная генерация

#### Many-to-One
```
In1 → [RNN] → [RNN] → [RNN]
In2 →          ↓
In3 →        Output
```
**Пример**: классификация текста, анализ настроения

#### Many-to-Many (синхронная)
```
In1 → [RNN] → [RNN] → [RNN]
       ↓       ↓       ↓
      Out1    Out2    Out3
```
**Пример**: разметка видео кадр за кадром

#### Many-to-Many (последовательность-последовательность)
```
Encoder:        Decoder:
In1 → [RNN] →  [RNN] → Out1
In2 →  [RNN] →  [RNN] → Out2
In3 →           [RNN] → Out3
```
**Пример**: машинный перевод, speech-to-text

### 3.3 Проблемы базового RNN

#### Проблема затухающих градиентов (Vanishing Gradients)

При обратном распространении через много временных шагов градиенты становятся очень маленькими:

```
∂L/∂W ∝ (∂hₜ/∂hₜ₋₁)ᵗ
```

Если |∂hₜ/∂hₜ₋₁| < 1, то градиент → 0 при большом t

**Последствия:**
- Сеть не учится на дальних зависимостях
- Теряется "долгая память"
- Обучение замедляется или останавливается

#### Проблема взрывающихся градиентов (Exploding Gradients)

Противоположная ситуация: градиенты → ∞

**Решение:**
- Gradient clipping (обрезка градиентов)
- Нормализация

---

## 4. Типы RNN (LSTM, GRU)

### 4.1 LSTM (Long Short-Term Memory)

**Идея**: добавить механизм **контроля потока информации**.

**Архитектура:**
```
        ┌─────────────┐
   Cₜ₋₁ │    Cell     │ Cₜ
   ────→│   State     │────→
        │             │
   hₜ₋₁─┤  ┌──────┐  ├─→ hₜ
        │  │Gates │  │
   xₜ──→│  └──────┘  │
        └─────────────┘
```

**Компоненты:**

1. **Cell State (Cₜ)** — долговременная память
   - "Конвейерная лента" информации

2. **Hidden State (hₜ)** — кратковременная память
   - Текущий выход

3. **Три гейта (Gates):**

   **a. Forget Gate (fₜ)** — что забыть из прошлого
   ```
   fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
   ```
   - Выход: [0, 1] для каждого элемента
   - 0 = полностью забыть, 1 = полностью сохранить

   **b. Input Gate (iₜ)** — что добавить нового
   ```
   iₜ = σ(Wi·[hₚₜ₋₁, xₜ] + bi)
   C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
   ```
   - iₜ: решает, сколько добавить
   - C̃ₜ: кандидаты на добавление

   **c. Output Gate (oₜ)** — что выдать на выход
   ```
   oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
   ```

**Обновление состояний:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
hₜ = oₜ ⊙ tanh(Cₜ)
```
где ⊙ — поэлементное умножение

**Пример работы:**
```
Предложение: "Кот, который был очень голодный, съел рыбу"

fₜ: забываем неважные детали ("был очень")
iₜ: запоминаем важное ("кот", "голодный", "съел", "рыбу")
oₜ: выдаем информацию для предсказания следующего слова
```

**Преимущества LSTM:**
✓ Решает проблему затухающих градиентов  
✓ Хранит долговременные зависимости  
✓ Контролирует поток информации  

**Недостатки:**
✗ Сложная архитектура  
✗ Много параметров (4 × больше, чем базовый RNN)  
✗ Медленное обучение  

### 4.2 GRU (Gated Recurrent Unit)

**Идея**: упрощенная версия LSTM с меньшим числом параметров.

**Архитектура:**
```
        ┌─────────────┐
   hₜ₋₁─┤  ┌──────┐  ├─→ hₜ
        │  │ Gates│  │
   xₜ──→│  └──────┘  │
        └─────────────┘
```

**Компоненты:**

1. **Reset Gate (rₜ)** — сколько забыть из прошлого
   ```
   rₜ = σ(Wr·[hₜ₋₁, xₜ] + br)
   ```

2. **Update Gate (zₜ)** — баланс между старым и новым
   ```
   zₜ = σ(Wz·[hₜ₋₁, xₜ] + bz)
   ```

3. **Кандидат на новое состояние**
   ```
   h̃ₜ = tanh(Wh·[rₜ⊙hₜ₋₁, xₜ] + bh)
   ```

**Обновление:**
```
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

**Интерпретация:**
- zₜ ≈ 1: использовать новую информацию
- zₜ ≈ 0: сохранить старую информацию

**Преимущества GRU:**
✓ Меньше параметров, чем LSTM (быстрее)  
✓ Проще архитектура  
✓ Часто сравнимое качество с LSTM  

**Недостатки:**
✗ Менее гибкая, чем LSTM  
✗ Для очень длинных последовательностей LSTM может быть лучше  

### 4.3 Сравнение RNN, LSTM, GRU

| Критерий | RNN | LSTM | GRU |
|----------|-----|------|-----|
| Параметры | ~N | ~4N | ~3N |
| Память | Кратковременная | Долговременная | Долговременная |
| Градиенты | Затухают | Стабильные | Стабильные |
| Скорость | Быстро | Медленно | Средне |
| Применение | Короткие последов. | Длинные последов. | Баланс |

**Когда что использовать:**
- **RNN**: простые задачи, короткие последовательности
- **LSTM**: сложные задачи, длинные зависимости, много данных
- **GRU**: когда нужен баланс между скоростью и качеством

### 4.4 Bidirectional RNN

**Идея**: обрабатывать последовательность в обоих направлениях.

**Архитектура:**
```
Forward:  → → → →
Input:    x₁ x₂ x₃ x₄
Backward: ← ← ← ←

Output: concat(forward, backward)
```

**Применение:**
- Когда доступна вся последовательность
- Контекст важен с обеих сторон
- Примеры: NLP, распознавание речи

**Недостаток**: нельзя использовать для онлайн-предсказаний (нужна вся последовательность)

---

## 5. Обучение RNN

### 5.1 Backpropagation Through Time (BPTT)

**Идея**: развернуть RNN во времени и применить обратное распространение.

**Алгоритм:**

1. **Forward pass**: вычислить выходы для всех временных шагов
   ```
   h₀ = 0
   for t in range(T):
       hₜ = f(Wxh·xₜ + Whh·hₜ₋₁ + bh)
       yₜ = g(Why·hₜ + by)
   ```

2. **Вычислить loss**:
   ```
   L = Σₜ L(yₜ, y_true_ₜ)
   ```

3. **Backward pass**: вычислить градиенты
   ```
   for t in range(T-1, -1, -1):
       ∂L/∂hₜ = ∂L/∂yₜ · ∂yₜ/∂hₜ + ∂L/∂hₜ₊₁ · ∂hₜ₊₁/∂hₜ
       ∂L/∂W = ... (chain rule)
   ```

4. **Обновить веса**:
   ```
   W := W - η∇L(W)
   ```

**Проблема**: для очень длинных последовательностей требует много памяти.

### 5.2 Truncated BPTT

**Решение**: разбивать последовательность на более короткие части.

```
Вместо:
[────────────T шагов────────────]

Делаем:
[──k шагов──][──k шагов──][──k шагов──]
```

**Плюсы:**
- Меньше памяти
- Быстрее обучение

**Минусы:**
- Теряются очень длинные зависимости

### 5.3 Оптимизаторы

#### SGD (Stochastic Gradient Descent)
```
W := W - η∇L(W)
```
- Простой, но медленный

#### Momentum
```
v := βv + η∇L(W)
W := W - v
```
- Ускоряет обучение в правильном направлении

#### Adam (Adaptive Moment Estimation)
```
m := β₁m + (1-β₁)∇L
v := β₂v + (1-β₂)(∇L)²
W := W - η·m/(√v + ε)
```
- Наиболее популярный
- Адаптивный learning rate для каждого параметра
- Обычно β₁=0.9, β₂=0.999

#### RMSprop
```
v := βv + (1-β)(∇L)²
W := W - η·∇L/√(v + ε)
```
- Хорошо работает для RNN

### 5.4 Регуляризация

#### Dropout
- Случайно "выключать" нейроны во время обучения
- Предотвращает переобучение
```python
keras.layers.Dropout(0.2)  # Отключить 20% нейронов
```

#### L2 Regularization (Weight Decay)
```
L_total = L + λΣw²
```
- Штрафует большие веса
- Предотвращает переобучение

#### Early Stopping
- Остановить обучение, когда validation loss перестает улучшаться

#### Gradient Clipping
```
if ||∇L|| > threshold:
    ∇L := threshold · ∇L / ||∇L||
```
- Предотвращает взрывающиеся градиенты

---

## 6. Биометрия и RNN

### 6.1 Что такое биометрия?

**Биометрия** — технология распознавания людей по их физиологическим или поведенческим характеристикам.

### 6.2 Типы биометрических данных

#### Статическая биометрия (Physiological)
Физические характеристики, которые не меняются со временем:

- **Отпечатки пальцев**
- **Радужная оболочка глаза**
- **Распознавание лица** (черты лица)
- **ДНК**
- **Геометрия ладони**
- **Сетчатка глаза**

**Особенности:**
- Стабильны во времени
- Сложно подделать
- Обычно используются CNN (сверточные сети)

#### Поведенческая биометрия (Behavioral)
Характеристики, связанные с поведением человека:

- **Динамика подписи** — скорость, давление, ускорение при написании
- **Походка** — манера ходьбы
- **Динамика нажатия клавиш** (Keystroke Dynamics) — ритм печати
- **Голос** — особенности речи, интонации
- **Движения мыши** — паттерны использования мыши
- **Использование смартфона** — как держишь, как свайпаешь

**Особенности:**
- Изменяются со временем
- Зависят от эмоционального состояния
- **Идеальны для RNN** (последовательные данные!)

### 6.3 Почему RNN для поведенческой биометрии?

**Пример: динамика подписи**

Подпись = последовательность точек (x, y, давление, время)

```
Точка 1: (x₁, y₁, p₁, t₁)
Точка 2: (x₂, y₂, p₂, t₂)
...
Точка n: (xₙ, yₙ, pₙ, tₙ)
```

**Важные особенности:**
- Порядок точек важен
- Скорость движения
- Паузы
- Давление на перо
- Ускорение

**RNN идеально подходит:**
- Обрабатывает последовательности переменной длины
- Учитывает временные зависимости
- Выявляет паттерны в динамике

**Архитектура для биометрии:**
```
Input: последовательность признаков
  ↓
[LSTM/GRU слои]
  ↓
[Dense слои]
  ↓
Output: вероятность принадлежности пользователю
```

### 6.4 Примеры задач

#### Задача 1: Аутентификация по подписи
```
Вход: динамика подписи
Выход: "настоящая" / "поддельная"
```

**Датасет:**
- 100 пользователей
- По 10 настоящих подписей от каждого
- По 20 поддельных подписей для каждого

**Архитектура:**
```python
model = Sequential([
    LSTM(128, input_shape=(sequence_length, num_features), return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Бинарная классификация
])
```

#### Задача 2: Распознавание по походке
```
Вход: последовательность 3D координат скелета
Выход: ID пользователя (многоклассовая классификация)
```

**Данные:**
- 30 кадров в секунду
- 25 суставов × 3 координаты = 75 признаков
- Последовательность из 60 кадров (2 секунды ходьбы)

**Архитектура:**
```python
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True), 
                  input_shape=(60, 75)),
    Dropout(0.4),
    Bidirectional(LSTM(128)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(num_users, activation='softmax')  # Многоклассовая
])
```

#### Задача 3: Keystroke Dynamics
```
Вход: время удержания клавиш + время между нажатиями
Выход: "легитимный пользователь" / "злоумышленник"
```

**Признаки:**
- Hold time: время удержания клавиши
- Flight time: время между отпусканием одной клавиши и нажатием следующей
- Digraph: паттерны из двух последовательных клавиш

---

## 7. Разделение данных и переобучение

### 7.1 Разделение данных

#### Train / Validation / Test Split

**Стандартное разделение:**
```
Train: 70%      Обучение модели
Validation: 15% Подбор гиперпараметров
Test: 15%       Финальная оценка
```

**Альтернативное (для больших данных):**
```
Train: 80%
Validation: 10%
Test: 10%
```

**Важно:**
- **Train** — обучаем модель, обновляем веса
- **Validation** — выбираем гиперпараметры, архитектуру, останавливаем обучение
- **Test** — **ТОЛЬКО** для финальной оценки, НЕ ТРОГАЕМ до конца!

#### K-Fold Cross-Validation

Для небольших датасетов:

```
Fold 1: [Val][────Train────]
Fold 2: [Train][Val][Train─]
Fold 3: [Train─][Val][Train]
Fold 4: [──Train][Val][─Tr─]
Fold 5: [─Train──────][Val]
```

**Алгоритм:**
1. Разделить данные на K частей
2. Для каждого fold:
   - Обучить модель на K-1 частях
   - Оценить на оставшейся части
3. Усреднить результаты

**Плюсы:**
- Используем все данные для обучения и валидации
- Более надежная оценка

**Минусы:**
- Требует K × больше времени на обучение

### 7.2 Переобучение (Overfitting)

#### Что это?

**Переобучение** — модель слишком хорошо запомнила обучающие данные, но плохо обобщает на новые.

**Признаки:**
```
Train Loss: ↓↓↓ (очень низкий)
Val Loss:   ↑↑↑ (растет или не улучшается)
```

**Визуально:**
```
Loss
  │
  │ ──── Val Loss
  │   ╱
  │  ╱
  │ ╱ ──── Train Loss
  └───────────────────→ Epochs
```

#### График переобучения

```python
import matplotlib.pyplot as plt

# История обучения
history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=100)

# График
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs Epoch')

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs Epoch')

plt.tight_layout()
plt.show()
```

**Интерпретация:**

1. **Нормальное обучение:**
   ```
   Train Loss ↓
   Val Loss ↓
   ```
   Модель обучается хорошо

2. **Переобучение:**
   ```
   Train Loss ↓↓↓
   Val Loss → или ↑
   ```
   Модель переобучилась после эпохи X

3. **Недообучение (Underfitting):**
   ```
   Train Loss → (высокий)
   Val Loss → (высокий)
   ```
   Модель слишком простая

#### Причины переобучения

1. Слишком сложная модель (много параметров)
2. Мало обучающих данных
3. Слишком долгое обучение
4. Отсутствие регуляризации
5. Шум в данных

#### Борьба с переобучением

**1. Больше данных**
- Data Augmentation (аугментация)
- Синтетические данные

**2. Упростить модель**
- Меньше слоев
- Меньше нейронов

**3. Регуляризация**
- Dropout
- L2 regularization
- Batch Normalization

**4. Early Stopping**
```python
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', 
                           patience=10,
                           restore_best_weights=True)

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          callbacks=[early_stop])
```

**5. Data Augmentation для временных рядов**
- Добавление шума
- Масштабирование
- Сдвиг во времени
- Изменение скорости

---

## 8. Метрики оценки

### 8.1 Матрица ошибок

```
                Predicted
              Positive | Negative
Actual  Pos      TP    |    FN
        Neg      FP    |    TN
```

**Пример:**
```
Задача: распознавание подписи
TP = 85: правильно распознаны настоящие подписи
TN = 90: правильно распознаны подделки
FP = 10: подделки, принятые за настоящие (Type I Error)
FN = 15: настоящие, принятые за подделки (Type II Error)
```

### 8.2 Основные метрики

#### Accuracy (Точность)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**Пример:** (85 + 90) / 200 = 0.875 = 87.5%

**Когда использовать:**
- Сбалансированные классы
- Все ошибки одинаково важны

**Не использовать:**
- Несбалансированные классы (например, 95% класс 0, 5% класс 1)

#### Precision (Точность положительных предсказаний)
```
Precision = TP / (TP + FP)
```
**Пример:** 85 / (85 + 10) = 0.894 = 89.4%

**Интерпретация:** Из всех предсказанных "настоящих" подписей, 89.4% действительно настоящие.

**Когда важна:**
- Когда False Positive дорого стоит
- Пример: медицинская диагностика (не хотим ложных тревог)

#### Recall (Полнота / Sensitivity / TPR)
```
Recall = TP / (TP + FN)
```
**Пример:** 85 / (85 + 15) = 0.85 = 85%

**Интерпретация:** Из всех настоящих подписей мы правильно распознали 85%.

**Когда важна:**
- Когда False Negative дорого стоит
- Пример: обнаружение мошенничества (не хотим пропустить мошенников)

#### F1-Score (Гармоническое среднее)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
**Пример:** 2 × (0.894 × 0.85) / (0.894 + 0.85) = 0.871 = 87.1%

**Интерпретация:** Баланс между Precision и Recall.

**Когда использовать:**
- Несбалансированные классы
- Нужен баланс между FP и FN

#### Specificity (TNR)
```
Specificity = TN / (TN + FP)
```
**Пример:** 90 / (90 + 10) = 0.90 = 90%

**Интерпретация:** Из всех подделок мы правильно распознали 90%.

### 8.3 ROC и AUC

#### TPR и FPR
```
TPR = TP / (TP + FN) = Recall
FPR = FP / (FP + TN) = 1 - Specificity
```

#### ROC-кривая
График зависимости TPR от FPR при изменении порога классификации.

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Получить вероятности
y_proba = model.predict(X_test)

# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

# График
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

#### AUC (Area Under Curve)
```
AUC ∈ [0, 1]
AUC = 1.0: идеальный классификатор
AUC = 0.5: случайный классификатор
```

**Интерпретация:**
- 0.9-1.0: отлично
- 0.8-0.9: очень хорошо
- 0.7-0.8: хорошо
- 0.6-0.7: удовлетворительно
- 0.5-0.6: плохо

### 8.4 Метрики регрессии

Если модель предсказывает непрерывные значения:

#### MSE (Mean Squared Error)
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```
**Чувствительна к выбросам**

#### MAE (Mean Absolute Error)
```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```
**Робастна к выбросам**

#### RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
**В тех же единицах, что и целевая переменная**

### 8.5 Многоклассовая классификация

#### Macro-averaged metrics
Среднее арифметическое метрик по всем классам:
```
Macro-F1 = (F1_class1 + F1_class2 + F1_class3) / 3
```

#### Micro-averaged metrics
Суммируем TP, FP, FN по всем классам:
```
Micro-Precision = Σ TP / (Σ TP + Σ FP)
```

#### Weighted-averaged metrics
Взвешенное среднее по размеру классов:
```
Weighted-F1 = Σ (n_class_i / n_total) × F1_class_i
```

### 8.6 Пример: полный отчет

```python
from sklearn.metrics import classification_report, confusion_matrix

# Предсказания
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

TP, FN = cm[1, 1], cm[1, 0]
FP, TN = cm[0, 1], cm[0, 0]

print(f"\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Метрики
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC
y_proba = model.predict(X_test)
auc = roc_auc_score(y_test, y_proba)
print(f"\nAUC: {auc:.3f}")
```

**Вывод:**
```
Confusion Matrix:
[[90 10]
 [15 85]]

TP: 85, TN: 90, FP: 10, FN: 15

Classification Report:
              precision    recall  f1-score   support
           0       0.86      0.90      0.88       100
           1       0.89      0.85      0.87       100
    accuracy                           0.88       200
   macro avg       0.88      0.88      0.88       200
weighted avg       0.88      0.88      0.88       200

AUC: 0.925
```

---

## 9. Практическая реализация

### 9.1 Подготовка данных для биометрии

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Пример: загрузка датасета подписей
# Каждая подпись = последовательность точек (x, y, pressure, timestamp)

def load_signature_data(file_path):
    """
    Загрузить данные подписей
    Формат: user_id, signature_id, timestamp, x, y, pressure
    """
    data = pd.read_csv(file_path)
    
    # Группировать по подписям
    signatures = []
    labels = []
    
    for (user_id, sig_id), group in data.groupby(['user_id', 'signature_id']):
        sequence = group[['x', 'y', 'pressure', 'timestamp']].values
        signatures.append(sequence)
        labels.append(user_id)
    
    return signatures, labels

# Загрузка
signatures, labels = load_signature_data('signatures.csv')

# Паддинг последовательностей до одинаковой длины
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = max(len(s) for s in signatures)
X = pad_sequences(signatures, maxlen=max_len, dtype='float32', padding='post')
y = np.array(labels)

# Нормализация признаков
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Разделение данных
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)  # 0.176 * 0.85 ≈ 0.15

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Sequence shape: {X_train.shape}")
```

### 9.2 Построение модели

```python
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Параметры
sequence_length = X_train.shape[1]
num_features = X_train.shape[2]
num_classes = len(np.unique(y))

# Архитектура
model = Sequential([
    # 1-й слой LSTM
    Bidirectional(LSTM(128, return_sequences=True), 
                  input_shape=(sequence_length, num_features)),
    Dropout(0.3),
    
    # 2-й слой LSTM
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    
    # 3-й слой LSTM
    LSTM(32),
    Dropout(0.3),
    
    # Полносвязные слои
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    
    # Выходной слой
    Dense(num_classes, activation='softmax')
])

# Компиляция
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### 9.3 Обучение модели

```python
# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Обучение
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
```

### 9.4 Визуализация обучения

```python
import matplotlib.pyplot as plt

# График loss и accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Loss vs Epoch', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Accuracy vs Epoch', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Определение точки переобучения
best_epoch = np.argmin(history.history['val_loss'])
print(f"Лучшая эпоха: {best_epoch + 1}")
print(f"Train Loss: {history.history['loss'][best_epoch]:.4f}")
print(f"Val Loss: {history.history['val_loss'][best_epoch]:.4f}")
```

### 9.5 Оценка на тестовой выборке

```python
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, roc_auc_score)
import seaborn as sns

# Предсказания
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Вычисление TP, TN, FP, FN для каждого класса
for i in range(num_classes):
    # Бинаризация: класс i vs остальные
    y_true_binary = (y_test == i).astype(int)
    y_pred_binary = (y_pred == i).astype(int)
    
    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()
    
    print(f"\nClass {i}:")
    print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"  Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.3f}")
    print(f"  Recall: {tp/(tp+fn) if (tp+fn)>0 else 0:.3f}")
```

### 9.6 ROC-кривая

```python
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Бинаризация меток для ROC
y_test_bin = label_binarize(y_test, classes=range(num_classes))

# ROC для каждого класса
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])

# Micro-average ROC
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), 
                                           y_pred_proba.ravel())
roc_auc["micro"] = roc_auc_score(y_test_bin, y_pred_proba, average='micro')

# График
plt.figure(figsize=(10, 8))

colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], 'k--', lw=2,
         label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curves - Multiclass', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Вывод AUC для всех классов
print("\n=== AUC Scores ===")
for i in range(num_classes):
    print(f"Class {i}: {roc_auc[i]:.3f}")
print(f"Micro-average: {roc_auc['micro']:.3f}")
```

### 9.7 Тестирование на реальных примерах

```python
# Загрузить несколько реальных примеров
def load_test_signature(file_path):
    """Загрузить одну тестовую подпись"""
    data = pd.read_csv(file_path)
    sequence = data[['x', 'y', 'pressure', 'timestamp']].values
    
    # Паддинг
    sequence_padded = pad_sequences([sequence], maxlen=max_len, 
                                    dtype='float32', padding='post')
    
    # Нормализация
    sequence_scaled = scaler.transform(
        sequence_padded.reshape(-1, num_features)
    ).reshape(sequence_padded.shape)
    
    return sequence_scaled

# Тестирование
test_files = [
    'test_signature_1.csv',
    'test_signature_2.csv',
    'test_signature_3.csv'
]

print("\n=== Testing on Real Examples ===")
for i, file in enumerate(test_files):
    signature = load_test_signature(file)
    prediction = model.predict(signature, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    print(f"\nSignature {i+1}: {file}")
    print(f"  Predicted User: {predicted_class}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  All probabilities: {prediction[0]}")
```

### 9.8 Сохранение и загрузка модели

```python
# Сохранение
model.save('biometric_rnn_model.h5')
print("Model saved successfully!")

# Сохранение архитектуры отдельно
model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model_json)

# Сохранение весов
model.save_weights('model_weights.h5')

# Загрузка
from tensorflow.keras.models import load_model

loaded_model = load_model('biometric_rnn_model.h5')
print("Model loaded successfully!")

# Проверка
test_loss, test_acc = loaded_model.evaluate(X_test, y_test, verbose=0)
print(f"Loaded model accuracy: {test_acc:.4f}")
```

---

## Заключение

**Основные выводы:**

1. **RNN** — мощный инструмент для последовательных данных
2. **LSTM и GRU** решают проблему долговременных зависимостей
3. **Биометрия** идеально подходит для RNN (поведенческие паттерны)
4. **Переобучение** — главный враг, требует регуляризации
5. **Правильная оценка** требует множества метрик (accuracy, precision, recall, F1, ROC-AUC)

**Рекомендации для лабораторной:**

1. Начните с простой архитектуры (1-2 слоя LSTM)
2. Постепенно усложняйте при необходимости
3. Всегда используйте Early Stopping
4. Визуализируйте процесс обучения
5. Тестируйте на реальных примерах
6. Анализируйте ошибки модели

**Дальнейшее изучение:**
- Attention механизмы
- Transformer архитектуры
- Transfer Learning для RNN
- Ансамбли моделей
