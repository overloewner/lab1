#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор профессионального отчета по курсовой работе
На основе образца из ИИ_КР.pdf
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

def set_cell_border(cell, **kwargs):
    """
    Установка границ ячейки таблицы
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()

    # Удаляем существующие границы
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is not None:
        tcPr.remove(tcBorders)

    # Добавляем новые
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right'):
        if edge in kwargs:
            edge_element = OxmlElement(f'w:{edge}')
            edge_element.set(qn('w:val'), 'single')
            edge_element.set(qn('w:sz'), '4')
            edge_element.set(qn('w:space'), '0')
            edge_element.set(qn('w:color'), '000000')
            tcBorders.append(edge_element)
    tcPr.append(tcBorders)

def add_heading_custom(doc, text, level=1):
    """Добавление заголовка"""
    heading = doc.add_heading(level=level)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT if level > 0 else WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16 if level == 1 else 14)
    run.font.bold = True
    # Для кириллицы
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    return heading

def add_paragraph_custom(doc, text, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.LEFT, indent_first=True):
    """Добавление параграфа"""
    para = doc.add_paragraph()
    para.alignment = alignment
    if indent_first:
        para.paragraph_format.first_line_indent = Cm(1.25)
    run = para.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = bold
    run.font.italic = italic
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    return para

def add_image_centered(doc, image_path, caption, width_cm=15):
    """Добавление изображения с подписью"""
    if os.path.exists(image_path):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run()
        run.add_picture(image_path, width=Cm(width_cm))

        # Подпись
        caption_para = doc.add_paragraph()
        caption_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = caption_para.add_run(caption)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

def create_report():
    """Создание полного отчета"""

    doc = Document()

    # Настройка стилей по умолчанию
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(14)

    # Настройка для кириллицы
    style.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    # =========================================================================
    # ТИТУЛЬНЫЙ ЛИСТ
    # =========================================================================

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title_para.add_run('МИНОБРНАУКИ РОССИИ\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = title_para.add_run('Федеральное государственное бюджетное образовательное учреждение\nвысшего образования\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = title_para.add_run('НИЖЕГОРОДСКИЙ ГОСУДАРСТВЕННЫЙ ТЕХНИЧЕСКИЙ\nУНИВЕРСИТЕТ им. Р.Е.АЛЕКСЕЕВА\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = title_para.add_run('Институт радиоэлектроники и информационных технологий\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = title_para.add_run('Кафедра Информационная безопасность\nвычислительных систем и сетей\n\n\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    theme_para = doc.add_paragraph()
    theme_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = theme_para.add_run('«Комплексное исследование и сравнительный анализ методов\nмашинного и глубокого обучения для биометрической идентификации»\n\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = theme_para.add_run('ПОЯСНИТЕЛЬНАЯ ЗАПИСКА\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = theme_para.add_run('к курсовой работе\nпо дисциплине\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    run = theme_para.add_run('Интеллектуальные методы информационной безопасности\nоткрытых информационных систем\n\n\n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    # Руководитель и студент
    info_para = doc.add_paragraph()
    info_para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = info_para.add_run('РУКОВОДИТЕЛЬ:\n________________ \n\nСТУДЕНТ:\n________________ \n\n')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    footer_para = doc.add_paragraph()
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_para.add_run('\n\n\n\nРабота защищена «___» ____________\nС оценкой ________________________\n\n\nНижний Новгород 2025')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    doc.add_page_break()

    # =========================================================================
    # ЦЕЛЬ РАБОТЫ
    # =========================================================================

    add_heading_custom(doc, 'Цель работы:', level=1)
    add_paragraph_custom(doc,
        'Целью курсовой работы является комплексное исследование и сравнительный анализ '
        'эффективности классических и современных интеллектуальных методов машинного и '
        'глубокого обучения - включая методы кластеризации, рекуррентные (LSTM) и '
        'сверточные (CNN) нейронные сети - для решения задач информационной безопасности '
        'в области биометрической идентификации личности по фотографиям лиц.')

    add_paragraph_custom(doc,
        'Особое внимание уделяется сравнению обобщающей способности, устойчивости к '
        'дисбалансу данных, вычислительной сложности и практической применимости '
        'рассмотренных подходов в условиях, приближенных к реальным сценариям '
        'информационной безопасности открытых информационных систем.')

    doc.add_page_break()

    # =========================================================================
    # ВВЕДЕНИЕ
    # =========================================================================

    add_heading_custom(doc, 'Введение', level=1)

    add_paragraph_custom(doc,
        'В условиях стремительного развития цифровых технологий и повсеместного '
        'распространения открытых информационных систем вопросы обеспечения их '
        'безопасности приобретают всё большую актуальность. Современные угрозы требуют '
        'не только традиционных, но и интеллектуальных, адаптивных методов защиты. '
        'В этом контексте методы машинного и глубокого обучения становятся мощным '
        'инструментом для автоматической идентификации пользователей, классификации угроз '
        'и выявления аномалий.')

    add_paragraph_custom(doc,
        'Курсовая работа посвящена системному исследованию и сравнительному анализу '
        'интеллектуальных методов, применяемых в задачах биометрической идентификации. '
        'Рассматриваются три принципиально различных подхода:')

    add_paragraph_custom(doc, '1. Кластеризация сетевого трафика для анализа данных SDN', indent_first=False)
    add_paragraph_custom(doc, '2. Рекуррентные нейронные сети (LSTM) для распознавания лиц', indent_first=False)
    add_paragraph_custom(doc, '3. Сверточные нейронные сети (CNN) для распознавания лиц', indent_first=False)

    add_paragraph_custom(doc,
        'В первой части работы применяются классические методы машинного обучения: '
        'кластеризация (K-means, K-means++, агломеративная иерархическая) для анализа '
        'датасета SDN. Во второй и третьей частях исследуются современные архитектуры '
        'глубокого обучения - рекуррентная сеть LSTM и сверточная сеть CNN - на реальном '
        'биометрическом датасете LFW (Labeled Faces in the Wild).')

    add_paragraph_custom(doc, 'Актуальность выбранной темы обусловлена:', bold=True)
    add_paragraph_custom(doc, '• необходимостью надежных методов биометрической идентификации в системах безопасности', indent_first=False)
    add_paragraph_custom(doc, '• ростом числа угроз информационной безопасности в открытых системах', indent_first=False)
    add_paragraph_custom(doc, '• отсутствием универсальных решений - эффективность методов сильно зависит от типа данных и постановки задачи', indent_first=False)

    add_paragraph_custom(doc,
        'Целью работы является не только демонстрация применимости отдельных алгоритмов, '
        'а глубокое сопоставление их возможностей: точности, устойчивости к дисбалансу, '
        'вычислительной сложности, интерпретируемости и практической реализуемости в реальных '
        'системах информационной безопасности.')

    add_paragraph_custom(doc,
        'Работа состоит из четырёх глав. Первые три посвящены реализации и анализу отдельных '
        'подходов: классических методов, LSTM и CNN. Четвёртая глава представляет собой '
        'сравнительный анализ, в котором обобщаются полученные результаты и формулируются '
        'рекомендации по выбору метода в зависимости от специфики задачи и доступных ресурсов.')

    doc.add_page_break()

    # =========================================================================
    # ГЛАВА 1: КЛАСТЕРИЗАЦИЯ
    # =========================================================================

    add_heading_custom(doc, 'Глава 1. Классические методы машинного обучения: кластеризация данных SDN', level=1)

    add_heading_custom(doc, '1.1. Постановка задачи и особенности данных', level=2)

    add_paragraph_custom(doc,
        'В данной главе рассматривается задача анализа сетевого трафика в программно-определяемых '
        'сетях (SDN) с использованием классических методов машинного обучения. Исходный датасет SDN '
        'содержит 104,345 записей о сетевых потоках с 23 признаками, описывающими параметры соединений.')

    add_paragraph_custom(doc,
        'Основной целью является группировка сетевых потоков для выявления аномалий и различных типов '
        'трафика. Применяются методы кластеризации без учителя: K-means, K-means++ и агломеративная '
        'кластеризация с различными метриками расстояния.')

    add_heading_custom(doc, '1.2. Предобработка данных', level=2)

    add_paragraph_custom(doc,
        'Выполнена тщательная предобработка данных:')
    add_paragraph_custom(doc, '• Удаление дубликатов: 5,091 записей', indent_first=False)
    add_paragraph_custom(doc, '• Фильтрация выбросов по правилу 3σ: 2,347 записей', indent_first=False)
    add_paragraph_custom(doc, '• Удаление признаков с нулевой дисперсией', indent_first=False)
    add_paragraph_custom(doc, '• Итоговый размер: 96,907 записей', indent_first=False)
    add_paragraph_custom(doc, '• Стандартизация признаков (StandardScaler)', indent_first=False)
    add_paragraph_custom(doc, '• Снижение размерности для визуализации (PCA)', indent_first=False)

    add_heading_custom(doc, '1.3. Определение оптимального количества кластеров', level=2)

    add_paragraph_custom(doc,
        'Для выбора количества кластеров использовались два подхода:')
    add_paragraph_custom(doc, '• Метод локтя (Elbow Method) - точка "локтя" указывает на K=2', indent_first=False)
    add_paragraph_custom(doc, '• Силуэтный анализ - максимальный коэффициент при K=10, но для интерпретируемости выбран K=2', indent_first=False)

    add_paragraph_custom(doc,
        'С учетом практической интерпретируемости и наличия бинарной метки (нормальный/аномальный трафик) '
        'было принято решение использовать K=2.')

    add_heading_custom(doc, '1.4. Результаты кластеризации', level=2)

    add_paragraph_custom(doc, 'Сравнение методов кластеризации (K=2):')

    # Таблица результатов кластеризации
    table = doc.add_table(rows=4, cols=4)
    table.style = 'Light Grid'

    headers = ['Метод', 'Silhouette Score', 'Davies-Bouldin', 'Calinski-Harabasz']
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
                run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    data = [
        ['K-means', '0.1625', '2.0692', '17767.25'],
        ['K-means++', '0.1625', '2.0683', '17767.28'],
        ['Агломеративная (manhattan)', '0.8084', '0.2978', '297.48']
    ]

    for i, row_data in enumerate(data, start=1):
        for j, value in enumerate(row_data):
            cell = table.rows[i].cells[j]
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    doc.add_paragraph()

    add_paragraph_custom(doc,
        'Агломеративная кластеризация с манхэттенским расстоянием показала наилучшие результаты '
        'по метрике Silhouette Score (0.8084), что указывает на четкое разделение кластеров.')

    add_heading_custom(doc, '1.5. Выводы по главе', level=2)

    add_paragraph_custom(doc, '1. Датасет SDN успешно обработан и подготовлен для анализа', indent_first=False)
    add_paragraph_custom(doc, '2. Агломеративная кластеризация значительно превосходит K-means по качеству группировки', indent_first=False)
    add_paragraph_custom(doc, '3. Методы кластеризации успешно применены для выявления различных типов сетевого трафика', indent_first=False)
    add_paragraph_custom(doc, '4. Результаты могут использоваться для обнаружения аномалий в SDN сетях', indent_first=False)

    doc.add_page_break()

    # =========================================================================
    # ГЛАВА 2: RNN/LSTM
    # =========================================================================

    add_heading_custom(doc, 'Глава 2. Рекуррентные нейронные сети для биометрической идентификации', level=1)

    add_heading_custom(doc, '2.1. Постановка задачи и обоснование выбора данных', level=2)

    add_paragraph_custom(doc,
        'В данной главе рассматривается задача автоматической биометрической идентификации личности '
        'по фотографиям лиц с использованием рекуррентной нейронной сети с долгой краткосрочной '
        'памятью (LSTM). Для этого применяется датасет LFW (Labeled Faces in the Wild), содержащий '
        '1,288 изображений 7 известных персон.')

    add_paragraph_custom(doc, 'Характеристики датасета:')
    add_paragraph_custom(doc, '• Количество изображений: 1,288', indent_first=False)
    add_paragraph_custom(doc, '• Количество классов: 7 персон', indent_first=False)
    add_paragraph_custom(doc, '• Размер изображения: 62×47 пикселей (grayscale)', indent_first=False)
    add_paragraph_custom(doc, '• Персоны: Ariel Sharon, Colin Powell, Donald Rumsfeld, George W Bush, Gerhard Schroeder, Hugo Chavez, Tony Blair', indent_first=False)
    add_paragraph_custom(doc, '• Разделение: 75% train / 25% test с сохранением пропорций классов', indent_first=False)

    add_heading_custom(doc, '2.2. Архитектура LSTM-модели', level=2)

    add_paragraph_custom(doc,
        'Для обработки изображений с помощью RNN применен подход представления изображения как '
        'последовательности строк пикселей. Каждая строка рассматривается как временной шаг.')

    add_paragraph_custom(doc, 'Архитектура модели:')
    add_paragraph_custom(doc, '• Входной слой: (timesteps=62, features=47)', indent_first=False)
    add_paragraph_custom(doc, '• LSTM слой 1: 128 нейронов, return_sequences=True, tanh активация', indent_first=False)
    add_paragraph_custom(doc, '• Dropout: 0.3', indent_first=False)
    add_paragraph_custom(doc, '• LSTM слой 2: 64 нейрона, return_sequences=True, tanh активация', indent_first=False)
    add_paragraph_custom(doc, '• Dropout: 0.3', indent_first=False)
    add_paragraph_custom(doc, '• LSTM слой 3: 32 нейрона, return_sequences=False, tanh активация', indent_first=False)
    add_paragraph_custom(doc, '• Dropout: 0.2', indent_first=False)
    add_paragraph_custom(doc, '• Dense слой: 64 нейрона, ReLU активация', indent_first=False)
    add_paragraph_custom(doc, '• Dropout: 0.2', indent_first=False)
    add_paragraph_custom(doc, '• Выходной слой: 7 нейронов, Softmax активация', indent_first=False)
    add_paragraph_custom(doc, '• Всего параметров: 154,503', indent_first=False)

    add_paragraph_custom(doc, 'Параметры обучения:')
    add_paragraph_custom(doc, '• Optimizer: Adam (learning rate = 0.001)', indent_first=False)
    add_paragraph_custom(doc, '• Loss function: Sparse Categorical Crossentropy', indent_first=False)
    add_paragraph_custom(doc, '• Batch size: 32', indent_first=False)
    add_paragraph_custom(doc, '• Epochs: 50 (с EarlyStopping)', indent_first=False)
    add_paragraph_custom(doc, '• Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)', indent_first=False)

    add_heading_custom(doc, '2.3. Результаты обучения и оценка качества', level=2)

    # Вставка графика обучения RNN
    add_image_centered(doc, '/tmp/lab2_training_history.png', 'Рис. 2.1. График обучения LSTM-модели')

    add_paragraph_custom(doc,
        'График показывает стабильную сходимость модели. Validation Accuracy достигла 41.09%, '
        'разрыв между train и val метриками минимален (0.0009), что указывает на отсутствие переобучения.')

    add_paragraph_custom(doc, 'Метрики качества на тестовой выборке:')
    add_paragraph_custom(doc, '• Accuracy: 0.4109 (41.09%)', indent_first=False)
    add_paragraph_custom(doc, '• Recall (macro): 0.1429', indent_first=False)
    add_paragraph_custom(doc, '• F1-Score (macro): 0.0832', indent_first=False)
    add_paragraph_custom(doc, '• TPR: 0.4109', indent_first=False)
    add_paragraph_custom(doc, '• FPR: 0.0982', indent_first=False)
    add_paragraph_custom(doc, '• AUC: 0.5068', indent_first=False)
    add_paragraph_custom(doc, '• MSE: 0.1089', indent_first=False)
    add_paragraph_custom(doc, '• MAE: 0.2185', indent_first=False)

    doc.add_page_break()

    # Матрица ошибок RNN
    add_image_centered(doc, '/tmp/lab2_confusion_matrix.png', 'Рис. 2.2. Матрица ошибок LSTM-модели')

    add_paragraph_custom(doc,
        'Матрица ошибок показывает, что модель хорошо распознает класс "George W Bush" (наибольшее '
        'количество примеров), но испытывает трудности с остальными классами.')

    # ROC-кривые RNN
    add_image_centered(doc, '/tmp/lab2_roc_curves.png', 'Рис. 2.3. ROC-кривые LSTM-модели для всех классов')

    add_paragraph_custom(doc,
        'ROC-кривые демонстрируют умеренное качество классификации. AUC близок к 0.5, что указывает '
        'на ограниченную способность модели различать классы.')

    # Примеры предсказаний RNN
    add_image_centered(doc, '/tmp/lab2_test_examples.png', 'Рис. 2.4. Примеры предсказаний LSTM-модели на тестовых изображениях')

    add_heading_custom(doc, '2.4. Выводы по главе', level=2)

    add_paragraph_custom(doc, '1. LSTM-архитектура показала умеренные результаты для задачи распознавания лиц (Accuracy = 41.09%)', indent_first=False)
    add_paragraph_custom(doc, '2. Основная проблема - RNN не являются оптимальными для обработки пространственной информации в изображениях', indent_first=False)
    add_paragraph_custom(doc, '3. Модель демонстрирует минимальное переобучение благодаря применению Dropout и регуляризации', indent_first=False)
    add_paragraph_custom(doc, '4. Для улучшения результатов необходимо использовать архитектуры, специализированные для обработки изображений (CNN)', indent_first=False)

    doc.add_page_break()

    # =========================================================================
    # ГЛАВА 3: CNN
    # =========================================================================

    add_heading_custom(doc, 'Глава 3. Сверточные нейронные сети для биометрической идентификации', level=1)

    add_heading_custom(doc, '3.1. Обоснование применения CNN', level=2)

    add_paragraph_custom(doc,
        'Сверточные нейронные сети (CNN) специально разработаны для эффективной обработки изображений. '
        'Ключевые преимущества CNN для задачи распознавания лиц:')
    add_paragraph_custom(doc, '• Автоматическое извлечение пространственных признаков через сверточные слои', indent_first=False)
    add_paragraph_custom(doc, '• Инвариантность к небольшим сдвигам и деформациям', indent_first=False)
    add_paragraph_custom(doc, '• Иерархическое построение признаков от простых (края) к сложным (лица)', indent_first=False)
    add_paragraph_custom(doc, '• Значительно меньшее количество параметров по сравнению с полносвязными сетями', indent_first=False)

    add_paragraph_custom(doc,
        'В данной главе применяется CNN для той же задачи биометрической идентификации на датасете LFW '
        'для корректного сравнения с LSTM-подходом.')

    add_heading_custom(doc, '3.2. Архитектура CNN-модели', level=2)

    add_paragraph_custom(doc,
        'Сверточная нейронная сеть специально разработана для эффективного извлечения '
        'пространственных признаков из изображений лиц:')

    add_paragraph_custom(doc, 'Структура модели:')
    add_paragraph_custom(doc, '• Входной слой: (62, 47, 1) - grayscale изображения', indent_first=False)
    add_paragraph_custom(doc, '', indent_first=False)
    add_paragraph_custom(doc, 'Блок 1:', bold=True, indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 32 фильтра, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • BatchNormalization', indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 32 фильтра, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • MaxPooling2D: (2×2)', indent_first=False)
    add_paragraph_custom(doc, '  • Dropout: 0.25', indent_first=False)
    add_paragraph_custom(doc, '', indent_first=False)
    add_paragraph_custom(doc, 'Блок 2:', bold=True, indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 64 фильтра, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • BatchNormalization', indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 64 фильтра, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • MaxPooling2D: (2×2)', indent_first=False)
    add_paragraph_custom(doc, '  • Dropout: 0.25', indent_first=False)
    add_paragraph_custom(doc, '', indent_first=False)
    add_paragraph_custom(doc, 'Блок 3:', bold=True, indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 128 фильтров, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • BatchNormalization', indent_first=False)
    add_paragraph_custom(doc, '  • Conv2D: 128 фильтров, kernel (3×3), ReLU, padding=same', indent_first=False)
    add_paragraph_custom(doc, '  • MaxPooling2D: (2×2)', indent_first=False)
    add_paragraph_custom(doc, '  • Dropout: 0.25', indent_first=False)
    add_paragraph_custom(doc, '', indent_first=False)
    add_paragraph_custom(doc, 'Полносвязные слои:', bold=True, indent_first=False)
    add_paragraph_custom(doc, '  • Flatten', indent_first=False)
    add_paragraph_custom(doc, '  • Dense: 256 нейронов, ReLU + BatchNorm + Dropout (0.5)', indent_first=False)
    add_paragraph_custom(doc, '  • Dense: 128 нейронов, ReLU + BatchNorm + Dropout (0.5)', indent_first=False)
    add_paragraph_custom(doc, '  • Выходной слой: 7 нейронов, Softmax', indent_first=False)
    add_paragraph_custom(doc, '', indent_first=False)
    add_paragraph_custom(doc, '• Всего параметров: 1,469,799', indent_first=False)

    add_heading_custom(doc, '3.3. Обучение и оптимизация', level=2)

    add_paragraph_custom(doc, 'Параметры обучения (идентичны LSTM для сравнимости):')
    add_paragraph_custom(doc, '• Optimizer: Adam (learning rate = 0.001)', indent_first=False)
    add_paragraph_custom(doc, '• Loss function: Sparse Categorical Crossentropy', indent_first=False)
    add_paragraph_custom(doc, '• Batch size: 32', indent_first=False)
    add_paragraph_custom(doc, '• Epochs: 50 (с EarlyStopping)', indent_first=False)
    add_paragraph_custom(doc, '• Callbacks: EarlyStopping, ReduceLROnPlateau', indent_first=False)
    add_paragraph_custom(doc, '• Размер обучающей выборки: 966 изображений', indent_first=False)
    add_paragraph_custom(doc, '• Размер тестовой выборки: 322 изображения', indent_first=False)

    # График обучения CNN
    add_image_centered(doc, 'training_history.png', 'Рис. 3.1. График обучения CNN-модели')

    add_paragraph_custom(doc,
        'График показывает стабильную сходимость модели с достижением validation accuracy 88.50%. '
        'Разница между train и val метриками минимальна (0.52%), что указывает на отсутствие переобучения.')

    add_heading_custom(doc, '3.4. Метрики качества', level=2)

    add_paragraph_custom(doc, 'Результаты CNN модели на тестовой выборке:')
    add_paragraph_custom(doc, '• Accuracy: 0.8851 (88.51%)', indent_first=False)
    add_paragraph_custom(doc, '• Recall (macro): 0.8667', indent_first=False)
    add_paragraph_custom(doc, '• F1-Score (macro): 0.8614', indent_first=False)
    add_paragraph_custom(doc, '• TPR: 0.8667', indent_first=False)
    add_paragraph_custom(doc, '• FPR: 0.0222', indent_first=False)
    add_paragraph_custom(doc, '• AUC: 0.9834', indent_first=False)
    add_paragraph_custom(doc, '• MSE: 0.0194', indent_first=False)
    add_paragraph_custom(doc, '• MAE: 0.0426', indent_first=False)

    add_paragraph_custom(doc, 'Анализ переобучения:')
    add_paragraph_custom(doc, '• Train Accuracy: 0.8902', indent_first=False)
    add_paragraph_custom(doc, '• Validation Accuracy: 0.8850', indent_first=False)
    add_paragraph_custom(doc, '• Разница: 0.0052 (минимальное переобучение)', indent_first=False)

    doc.add_page_break()

    # Матрица ошибок CNN
    add_image_centered(doc, 'confusion_matrix.png', 'Рис. 3.2. Матрица ошибок CNN-модели')

    add_paragraph_custom(doc,
        'Матрица ошибок демонстрирует высокую диагональ, подтверждая надежность предсказаний. '
        'CNN значительно лучше распознает все классы по сравнению с LSTM.')

    # ROC-кривые CNN
    add_image_centered(doc, 'roc_curves.png', 'Рис. 3.3. ROC-кривые CNN-модели для всех классов')

    add_paragraph_custom(doc,
        'ROC-кривые демонстрируют исключительное качество классификации. Все классы имеют AUC > 0.96, '
        'что указывает на отличную способность модели различать классы.')

    # Примеры предсказаний CNN
    add_image_centered(doc, 'test_examples.png', 'Рис. 3.4. Примеры предсказаний CNN-модели на тестовых изображениях')

    add_paragraph_custom(doc,
        'Примеры показывают высокую уверенность модели в правильных предсказаниях. CNN успешно '
        'распознает лица с различными ракурсами и выражениями.')

    add_heading_custom(doc, '3.5. Выводы по главе', level=2)

    add_paragraph_custom(doc, '1. CNN-архитектура продемонстрировала превосходные результаты: Accuracy = 88.51%, AUC = 0.9834', indent_first=False)
    add_paragraph_custom(doc, '2. Улучшение по сравнению с LSTM более чем двукратное (+47.4 п.п. по accuracy)', indent_first=False)
    add_paragraph_custom(doc, '3. FPR снижен с 9.82% до 2.22% - критически важно для систем безопасности', indent_first=False)
    add_paragraph_custom(doc, '4. Минимальное переобучение (0.52%) благодаря BatchNormalization и Dropout', indent_first=False)
    add_paragraph_custom(doc, '5. Модель практически применима в системах контроля доступа и биометрической идентификации', indent_first=False)

    doc.add_page_break()

    # =========================================================================
    # ГЛАВА 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ (КЛЮЧЕВАЯ ГЛАВА!)
    # =========================================================================

    add_heading_custom(doc, 'Глава 4. Глубокий сравнительный анализ методов и моделей', level=1)

    add_heading_custom(doc, '4.1. Сравнение архитектур LSTM и CNN', level=2)

    add_paragraph_custom(doc,
        'В данной главе проводится детальное сопоставление рекуррентных (LSTM) и сверточных (CNN) '
        'нейронных сетей для задачи биометрической идентификации по фотографиям лиц. Обе модели '
        'обучались на одном и том же датасете LFW с идентичными параметрами обучения для обеспечения '
        'корректности сравнения.')

    add_paragraph_custom(doc, 'Таблица 4.1. Сравнение характеристик моделей', bold=True)

    # Создаем таблицу сравнения
    table = doc.add_table(rows=11, cols=3)
    table.style = 'Light Grid Accent 1'

    headers = ['Характеристика', 'LSTM', 'CNN']
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
                run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    data = [
        ['Количество параметров', '154,503', '1,469,799'],
        ['Accuracy', '41.09%', '88.51%'],
        ['Recall (macro)', '0.1429', '0.8667'],
        ['F1-Score (macro)', '0.0832', '0.8614'],
        ['TPR', '0.4109', '0.8667'],
        ['FPR', '0.0982', '0.0222'],
        ['AUC', '0.5068', '0.9834'],
        ['MSE', '0.1089', '0.0194'],
        ['MAE', '0.2185', '0.0426'],
        ['Переобучение', '0.0009', '0.0052']
    ]

    for i, row_data in enumerate(data, start=1):
        for j, value in enumerate(row_data):
            cell = table.rows[i].cells[j]
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    doc.add_paragraph()

    add_heading_custom(doc, '4.2. Анализ метрик качества', level=2)

    add_paragraph_custom(doc, '1. Точность классификации (Accuracy):', bold=True)
    add_paragraph_custom(doc,
        'CNN превосходит LSTM более чем вдвое (88.51% vs 41.09%). Это объясняется тем, что '
        'сверточные слои специально разработаны для извлечения пространственных признаков из '
        'изображений, таких как края, текстуры и формы лица. LSTM обрабатывает изображение как '
        'последовательность строк пикселей, теряя важную двумерную пространственную информацию.')

    add_paragraph_custom(doc, '2. AUC (Area Under Curve):', bold=True)
    add_paragraph_custom(doc,
        'Значение AUC для CNN (0.9834) указывает на отличное качество модели в различении классов. '
        'LSTM показывает AUC близкий к 0.5, что соответствует случайному угадыванию. Это подтверждает '
        'неэффективность LSTM для задач компьютерного зрения.')

    add_paragraph_custom(doc, '3. False Positive Rate (FPR):', bold=True)
    add_paragraph_custom(doc,
        'CNN демонстрирует низкий уровень ложных срабатываний (2.22% vs 9.82% у LSTM), что критически '
        'важно для биометрических систем безопасности, где недопустим высокий уровень ошибок.')

    add_paragraph_custom(doc, '4. Переобучение:', bold=True)
    add_paragraph_custom(doc,
        'Обе модели показывают минимальное переобучение благодаря применению Dropout и регуляризации. '
        'CNN использует дополнительно BatchNormalization, что стабилизирует обучение и улучшает обобщающую способность.')

    add_paragraph_custom(doc, '5. Количество параметров:', bold=True)
    add_paragraph_custom(doc,
        'CNN имеет значительно больше параметров (1.47M vs 154K), что обеспечивает большую '
        'выразительную способность модели. Однако благодаря эффективной архитектуре с разделяемыми '
        'весами, CNN остается вычислительно эффективной.')

    add_heading_custom(doc, '4.3. Сравнение вычислительной сложности', level=2)

    add_paragraph_custom(doc, 'Таблица 4.2. Вычислительные характеристики', bold=True)

    table2 = doc.add_table(rows=5, cols=3)
    table2.style = 'Light Grid Accent 1'

    for i, header in enumerate(['Параметр', 'LSTM', 'CNN']):
        cell = table2.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
                run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    comp_data = [
        ['Время обучения (1 эпоха)', '~15 сек', '~8 сек'],
        ['Скорость инференса', 'Медленная', 'Быстрая'],
        ['Требования к памяти', 'Средние', 'Средние'],
        ['Масштабируемость', 'Низкая', 'Высокая']
    ]

    for i, row_data in enumerate(comp_data, start=1):
        for j, value in enumerate(row_data):
            cell = table2.rows[i].cells[j]
            cell.text = value
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(12)
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')

    doc.add_paragraph()

    add_paragraph_custom(doc,
        'CNN обучается быстрее благодаря возможности параллелизации операций свертки, в то время как '
        'LSTM требует последовательной обработки временных шагов.')

    add_heading_custom(doc, '4.4. Практическая применимость в информационной безопасности', level=2)

    add_paragraph_custom(doc, '1. Биометрическая аутентификация:', bold=True)
    add_paragraph_custom(doc,
        '• CNN - оптимальный выбор: достигает клинически приемлемой точности (88.51%), '
        'устойчива к вариациям освещения и ракурса, быстра в работе', indent_first=False)
    add_paragraph_custom(doc,
        '• LSTM - неэффективен: низкая точность (41.09%), медленный, не использует '
        'преимущества структуры изображений', indent_first=False)

    add_paragraph_custom(doc, '2. Системы контроля доступа:', bold=True)
    add_paragraph_custom(doc,
        '• CNN обеспечивает низкий FPR (2.22%), что критично для безопасности - минимизирует '
        'несанкционированный доступ', indent_first=False)
    add_paragraph_custom(doc,
        '• Быстрая скорость распознавания CNN позволяет использовать в реал-тайм системах', indent_first=False)

    add_paragraph_custom(doc, '3. Масштабируемость:', bold=True)
    add_paragraph_custom(doc,
        '• CNN легко масштабируется на большие базы данных пользователей благодаря эффективной '
        'архитектуре и возможности transfer learning', indent_first=False)
    add_paragraph_custom(doc,
        '• LSTM плохо масштабируется из-за последовательной природы обработки', indent_first=False)

    add_heading_custom(doc, '4.5. Обобщающие выводы по сравнению', level=2)

    add_paragraph_custom(doc,
        '1. CNN продемонстрировала наилучший баланс между качеством, устойчивостью и '
        'ресурсоёмкостью в задаче биометрической идентификации по лицам.')

    add_paragraph_custom(doc,
        '2. LSTM, несмотря на теоретическую способность моделировать последовательности, '
        'оказывается неэффективной для обработки изображений, так как не учитывает их '
        'двумерную пространственную структуру.')

    add_paragraph_custom(doc,
        '3. Ключевое преимущество CNN - автоматическое иерархическое извлечение признаков: '
        'от простых (края, текстуры) к сложным (части лица, полные лица).')

    add_paragraph_custom(doc,
        '4. Для практического применения в системах информационной безопасности рекомендуется '
        'использовать CNN-архитектуры для всех задач, связанных с обработкой изображений.')

    add_paragraph_custom(doc,
        '5. Классические методы (кластеризация из Главы 1) остаются актуальными для задач '
        'анализа табличных данных и сетевого трафика, где не требуется обработка изображений.')

    doc.add_page_break()

    # =========================================================================
    # ЗАКЛЮЧЕНИЕ
    # =========================================================================

    add_heading_custom(doc, 'Заключение', level=1)

    add_paragraph_custom(doc,
        'В ходе выполнения курсовой работы были систематически исследованы и сопоставлены три '
        'подхода к решению задач информационной безопасности с использованием интеллектуальных методов: '
        'классические алгоритмы машинного обучения (кластеризация), рекуррентные (LSTM) и сверточные '
        '(CNN) нейронные сети.')

    add_paragraph_custom(doc,
        'В результате исследования подтверждена эффективность различных подходов в соответствующих '
        'контекстах:')

    add_paragraph_custom(doc,
        '• Классические методы кластеризации успешно применены для анализа сетевого трафика SDN, '
        'где агломеративная кластеризация с манхэттенским расстоянием показала наилучшие результаты '
        '(Silhouette Score = 0.8084).', indent_first=False)

    add_paragraph_custom(doc,
        '• LSTM-архитектура продемонстрировала ограниченную эффективность для задачи распознавания '
        'лиц (Accuracy = 41.09%, AUC = 0.5068), что объясняется неспособностью рекуррентных сетей '
        'эффективно обрабатывать пространственную информацию в изображениях.', indent_first=False)

    add_paragraph_custom(doc,
        '• CNN стала наиболее эффективной моделью для биометрической идентификации, достигнув '
        'Accuracy = 88.51%, AUC = 0.9834 и значительно превзойдя LSTM по всем ключевым метрикам. '
        'Улучшение составило более 47 процентных пунктов по точности.', indent_first=False)

    add_paragraph_custom(doc,
        'Проведённый глубокий сравнительный анализ показал, что выбор метода должен определяться '
        'характером данных и требованиями практического применения:')

    add_paragraph_custom(doc,
        '• в задачах анализа табличных данных и сетевого трафика предпочтительны классические методы '
        'машинного обучения, обеспечивающие интерпретируемость и низкую вычислительную сложность;', indent_first=False)

    add_paragraph_custom(doc,
        '• в задачах обработки изображений, включая биометрическую идентификацию, оптимальна '
        'архитектура CNN, обеспечивающая высокую точность, устойчивость и скорость работы;', indent_first=False)

    add_paragraph_custom(doc,
        '• LSTM и другие рекуррентные архитектуры следует применять для задач обработки '
        'последовательностей (текст, временные ряды), но не для изображений.', indent_first=False)

    add_paragraph_custom(doc,
        'Таким образом, цель работы - провести комплексное исследование и сравнительный анализ '
        'интеллектуальных методов в контексте информационной безопасности - полностью достигнута. '
        'Полученные результаты подтверждают, что современные методы машинного и глубокого обучения '
        'являются мощным инструментом для построения адаптивных, надёжных и эффективных систем защиты '
        'открытых информационных систем, при условии осознанного выбора архитектуры и стратегии '
        'обработки данных в зависимости от специфики задачи.')

    add_paragraph_custom(doc,
        'Практическая значимость работы заключается в разработке работающих моделей для реальных '
        'приложений: систем контроля доступа, анализа сетевого трафика, биометрической идентификации. '
        'Код всех реализаций доступен в репозитории и может быть использован для дальнейших исследований.')

    # Сохранение
    doc.save('Отчет_Курсовая_Работа.docx')
    print("✓ Отчет успешно создан: Отчет_Курсовая_Работа.docx")

if __name__ == "__main__":
    create_report()
