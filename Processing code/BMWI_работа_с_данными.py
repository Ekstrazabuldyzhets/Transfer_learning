#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка датасета BMW i3 для трансферного обучения

Этот скрипт:
1. Загружает CSV файлы из папки Measurement Data
2. Приводит данные к единому формату, аналогичному лабораторным датасетам
3. Выполняет очистку от артефактов CAN-шины и пропусков
4. Сохраняет обработанные данные в структурированном виде
5. Подготавливает данные для использования в трансферном обучении

Особенности:
- SoC от BMS используется как целевая переменная (не эталонная)
- Сохраняются только колонки, необходимые для прогнозирования SoC
- Категория A и B обрабатываются отдельно
- Для категории B сохраняются данные о нагревателе (опционально)
"""

import os
import warnings
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ============================================================================

# Имена колонок в выходных данных (унифицированный формат)
# Аналогично лабораторным датасетам Panasonic и LG HG2
COLUMNS_OUT = [
    "Time",  # время в секундах
    "Voltage",  # напряжение в вольтах
    "Current",  # ток в амперах (+ заряд, - разряд)
    "Temperature",  # температура батареи в °C
    "Velocity",  # скорость в км/ч
    "SoC_BMS",  # состояние заряда от BMS (в долях 0-1)
    "SoC_BMS_percent"  # состояние заряда от BMS (в процентах)
]

# Дополнительные колонки для категории B (опционально)
COLUMNS_B_EXTRA = [
    "Heating_Power_kW",  # мощность PTC-нагревателя в кВт
    "Ambient_Temperature",  # температура окружающего воздуха в °C
    "Throttle",  # положение педали газа в %
    "Motor_Torque"  # момент электродвигателя в Н·м
]

# Колонки, которые нужно сохранить из исходных файлов категории A
KEEP_COLS_A = [
    'Time [s]',
    'Battery Voltage [V]',
    'Battery Current [A]',
    'Battery Temperature [°C]',
    'Velocity [km/h]',
    'SoC [%]'
]

# Колонки, которые нужно сохранить из исходных файлов категории B
KEEP_COLS_B = [
    'Time [s]',
    'Battery Voltage [V]',
    'Battery Current [A]',
    'Battery Temperature [°C]',
    'Velocity [km/h]',
    'SoC [%]',
    'Heating Power CAN [kW]',
    'Ambient Temperature [°C]',
    'Throttle [%]',
    'Motor Torque [Nm]'
]

# Параметры чтения CSV
CSV_SEP = ';'
CSV_ENCODING = 'latin1'


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует все колонки DataFrame из строк в числа.

    В файлах BMW i3 десятичным разделителем является запятая,
    поэтому перед преобразованием заменяем запятые на точки.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными в строковом формате

    Returns
    -------
    pd.DataFrame
        DataFrame с числовыми колонками
    """
    df_result = df.copy()

    for col in df_result.columns:
        try:
            # Заменяем запятую на точку и преобразуем в число
            df_result[col] = pd.to_numeric(
                df_result[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )
        except Exception:
            continue

    return df_result


def clean_soc(df: pd.DataFrame, soc_col: str = 'SoC [%]') -> pd.DataFrame:
    """
    Очищает колонку состояния заряда от выбросов и пропусков.

    Особенности:
    - Удаляет строки с пропущенными значениями SoC (обычно в начале файла)
    - Ограничивает SoC диапазоном [0, 100]
    - Заполняет единичные пропуски линейной интерполяцией

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными
    soc_col : str
        Название колонки с состоянием заряда

    Returns
    -------
    pd.DataFrame
        DataFrame с очищенным SoC
    """
    if soc_col not in df.columns:
        return df

    # Удаляем строки с пропущенным SoC (BMS ещё не инициализирована)
    df_clean = df.dropna(subset=[soc_col]).copy()

    if len(df_clean) == 0:
        return df_clean

    # Ограничиваем SoC диапазоном [0, 100]
    df_clean[soc_col] = df_clean[soc_col].clip(0, 100)

    # Интерполяция единичных пропусков (если остались)
    if df_clean[soc_col].isnull().any():
        df_clean[soc_col] = df_clean[soc_col].interpolate(method='linear', limit=5)

    return df_clean


def filter_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрует артефакты CAN-шины.

    Обрабатывает:
    - Выбросы мощности нагревателя (> 10 кВт считаем артефактом)
    - Резкие скачки тока (более 500 А считаем артефактом)
    - Выбросы напряжения вне физического диапазона (200-450 В)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными

    Returns
    -------
    pd.DataFrame
        DataFrame с отфильтрованными артефактами
    """
    df_filtered = df.copy()

    # Фильтрация напряжения (физический диапазон для BMW i3: 250-420 В)
    if 'Battery Voltage [V]' in df_filtered.columns:
        mask_voltage = (df_filtered['Battery Voltage [V]'] >= 250) & \
                       (df_filtered['Battery Voltage [V]'] <= 420)
        df_filtered = df_filtered[mask_voltage]

    # Фильтрация температуры (физический диапазон: -30°C ... +50°C)
    if 'Battery Temperature [°C]' in df_filtered.columns:
        mask_temp = (df_filtered['Battery Temperature [°C]'] >= -30) & \
                    (df_filtered['Battery Temperature [°C]'] <= 50)
        df_filtered = df_filtered[mask_temp]

    # Фильтрация тока (физический диапазон: -500 ... +200 А)
    if 'Battery Current [A]' in df_filtered.columns:
        mask_current = (df_filtered['Battery Current [A]'] >= -500) & \
                       (df_filtered['Battery Current [A]'] <= 200)
        df_filtered = df_filtered[mask_current]

    # Фильтрация скорости (физический диапазон: 0 ... 200 км/ч)
    if 'Velocity [km/h]' in df_filtered.columns:
        mask_velocity = (df_filtered['Velocity [km/h]'] >= 0) & \
                        (df_filtered['Velocity [km/h]'] <= 200)
        df_filtered = df_filtered[mask_velocity]

    # Фильтрация мощности нагревателя (артефакты > 10 кВт)
    if 'Heating Power CAN [kW]' in df_filtered.columns:
        mask_heater = (df_filtered['Heating Power CAN [kW]'] >= 0) & \
                      (df_filtered['Heating Power CAN [kW]'] <= 10)
        df_filtered = df_filtered[mask_heater]

    return df_filtered


def get_category(filename: str) -> str:
    """
    Определяет категорию файла (A или B) по имени.

    Parameters
    ----------
    filename : str
        Имя файла

    Returns
    -------
    str
        'A', 'B' или 'unknown'
    """
    if 'TripA' in filename:
        return 'A'
    elif 'TripB' in filename:
        return 'B'
    else:
        return 'unknown'


def load_bmw_csv(filepath: str) -> pd.DataFrame | None:
    """
    Загружает CSV файл BMW i3 и преобразует в унифицированный формат.

    Parameters
    ----------
    filepath : str
        Путь к CSV файлу

    Returns
    -------
    pd.DataFrame or None
        DataFrame в унифицированном формате или None при ошибке
    """
    try:
        # Читаем CSV как строки
        df_raw = pd.read_csv(filepath, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str)

        # Преобразуем в числа
        df_numeric = convert_to_numeric(df_raw)

        # Определяем категорию и выбираем колонки
        filename = os.path.basename(filepath)
        category = get_category(filename)

        if category == 'A':
            # Категория A: только основные колонки
            available_cols = [col for col in KEEP_COLS_A if col in df_numeric.columns]
            df_out = df_numeric[available_cols].copy()
        elif category == 'B':
            # Категория B: основные колонки + дополнительные
            available_cols = [col for col in KEEP_COLS_B if col in df_numeric.columns]
            df_out = df_numeric[available_cols].copy()
        else:
            return None

        # Переименовываем колонки в унифицированный формат
        rename_map = {
            'Time [s]': 'Time',
            'Battery Voltage [V]': 'Voltage',
            'Battery Current [A]': 'Current',
            'Battery Temperature [°C]': 'Temperature',
            'Velocity [km/h]': 'Velocity',
            'SoC [%]': 'SoC_BMS_percent',
            'Heating Power CAN [kW]': 'Heating_Power_kW',
            'Ambient Temperature [°C]': 'Ambient_Temperature',
            'Throttle [%]': 'Throttle',
            'Motor Torque [Nm]': 'Motor_Torque'
        }

        df_out = df_out.rename(columns=rename_map)

        # Добавляем SoC в долях
        if 'SoC_BMS_percent' in df_out.columns:
            df_out['SoC_BMS'] = df_out['SoC_BMS_percent'] / 100.0

        # Очистка SoC
        df_out = clean_soc(df_out, soc_col='SoC_BMS_percent')

        if len(df_out) == 0:
            return None

        # Фильтрация артефактов
        # Для фильтрации нужны исходные названия колонок, поэтому временно восстанавливаем
        temp_map = {v: k for k, v in rename_map.items()}
        temp_map['Heating_Power_kW'] = 'Heating Power CAN [kW]'
        temp_map['Ambient_Temperature'] = 'Ambient Temperature [°C]'

        df_for_filter = df_out.rename(columns=temp_map)
        df_filtered = filter_artifacts(df_for_filter)

        # Возвращаем в унифицированный формат
        df_result = df_filtered.rename(columns={v: k for k, v in temp_map.items()})

        return df_result

    except Exception as e:
        print(f"Ошибка загрузки {os.path.basename(filepath)}: {e}")
        return None


# ============================================================================
# ОСНОВНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ============================================================================

def process_bmw_file(filepath: str, output_dir: str, save_extra_cols: bool = True):
    """
    Обрабатывает один файл BMW i3 и сохраняет результат.

    Parameters
    ----------
    filepath : str
        Путь к исходному CSV файлу
    output_dir : str
        Директория для сохранения обработанного файла
    save_extra_cols : bool
        Сохранять ли дополнительные колонки (категория B)
    """
    filename = os.path.basename(filepath)
    category = get_category(filename)

    # Загрузка и обработка
    df = load_bmw_csv(filepath)

    if df is None or len(df) == 0:
        return

    # Формирование выходного набора колонок
    output_cols = COLUMNS_OUT.copy()

    if category == 'B' and save_extra_cols:
        extra_cols = [col for col in COLUMNS_B_EXTRA if col in df.columns]
        output_cols.extend(extra_cols)

    # Выбираем только нужные колонки
    available_cols = [col for col in output_cols if col in df.columns]
    df_out = df[available_cols].copy()

    # Нормализация времени (приводим к началу отсчёта с нуля)
    if 'Time' in df_out.columns:
        df_out['Time'] = df_out['Time'] - df_out['Time'].iloc[0]

    # Создание выходной директории
    os.makedirs(output_dir, exist_ok=True)

    # Сохранение
    output_filename = filename.replace('.csv', '_processed.csv')
    output_path = os.path.join(output_dir, output_filename)
    df_out.to_csv(output_path, index=False)


# ============================================================================
# ФУНКЦИЯ СБОРА ФАЙЛОВ
# ============================================================================

def collect_csv_files(source_dir: str) -> list:
    """
    Собирает все CSV файлы из директории.

    Parameters
    ----------
    source_dir : str
        Путь к папке с исходными данными

    Returns
    -------
    list
        Список путей к CSV файлам
    """
    files = []
    for f in os.listdir(source_dir):
        if f.endswith('.csv'):
            files.append(os.path.join(source_dir, f))
    return files


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main(source_directory: str, output_directory: str, save_extra_cols: bool = True):
    """
    Главная функция обработки датасета BMW i3.

    Parameters
    ----------
    source_directory : str
        Путь к папке с исходными CSV файлами
    output_directory : str
        Путь для сохранения обработанных файлов
    save_extra_cols : bool
        Сохранять ли дополнительные колонки для категории B
    """
    print("=" * 80)
    print("🔋 ОБРАБОТКА ДАТАСЕТА BMW i3")
    print("=" * 80)

    # Проверка существования директории
    if not os.path.exists(source_directory):
        print(f"\n❌ Ошибка: Исходная директория не найдена!")
        print(f"   Путь: {source_directory}")
        return

    print(f"\n📁 Исходная директория: {source_directory}")
    print(f"📁 Выходная директория: {output_directory}")

    # Создание выходной директории
    os.makedirs(output_directory, exist_ok=True)

    # Сбор файлов
    csv_files = collect_csv_files(source_directory)
    print(f"\n📂 Найдено CSV файлов: {len(csv_files)}")

    if len(csv_files) == 0:
        print("\n❌ CSV файлы не найдены!")
        return

    # Подсчёт категорий
    cat_a_count = sum(1 for f in csv_files if 'TripA' in f)
    cat_b_count = sum(1 for f in csv_files if 'TripB' in f)
    print(f"   Категория A (лето): {cat_a_count} файлов")
    print(f"   Категория B (зима): {cat_b_count} файлов")

    # Обработка файлов
    print("\n=== Обработка файлов ===")

    successful = 0
    failed = 0

    for filepath in tqdm(csv_files, desc="Обработка CSV файлов"):
        try:
            process_bmw_file(filepath, output_directory, save_extra_cols)
            successful += 1
        except Exception as e:
            print(f"   Ошибка обработки {os.path.basename(filepath)}: {e}")
            failed += 1

    # Итоги
    print("\n" + "=" * 80)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)

    print(f"\n📊 Статистика:")
    print(f"   - Исходных файлов: {len(csv_files)}")
    print(f"   - Успешно обработано: {successful}")
    print(f"   - Ошибок: {failed}")
    print(f"   - Категория A: {cat_a_count} файлов")
    print(f"   - Категория B: {cat_b_count} файлов")

    print(f"\n📁 Результаты сохранены в: {output_directory}")

    # Вывод информации о сохранённых колонках
    print(f"\n📋 Сохранённые колонки:")
    print(f"   Основные: {', '.join(COLUMNS_OUT)}")
    if save_extra_cols:
        print(f"   Дополнительные (категория B): {', '.join(COLUMNS_B_EXTRA)}")

    print("\n" + "=" * 80)


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == '__main__':
    # ========================================================================
    # НАСТРОЙКА ПУТЕЙ (ИЗМЕНИТЕ ПОД ВАШУ СТРУКТУРУ)
    # ========================================================================

    # Путь к исходным данным BMW i3
    SOURCE_DIRECTORY = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/Measurement Data"

    # Путь для сохранения обработанных данных
    OUTPUT_DIRECTORY = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/BMW_processed"

    # Флаг сохранения дополнительных колонок для категории B
    # True - сохранять данные о нагревателе, оборотах, педали газа
    # False - сохранять только основные колонки (время, напряжение, ток, температура, скорость, SoC)
    SAVE_EXTRA_COLS = True

    main(SOURCE_DIRECTORY, OUTPUT_DIRECTORY, SAVE_EXTRA_COLS)