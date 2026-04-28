#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обработка датасета Panasonic 18650PF Li-ion Battery Data
Структурированный код, аналогичный вашему стилю, с подробными пояснениями

Этот скрипт:
1. Загружает все .mat файлы из структуры датасета Panasonic
2. Извлекает данные из структуры 'meas'
3. Рассчитывает эталонный SoC из поля Ah
4. Использует C20 тест (только из папки 25°C) для калибровки
5. Выполняет кулонометрический расчёт SoC для всех типов файлов
6. Сохраняет обработанные данные в CSV файлы с сохранением структуры папок
"""

import os
import warnings
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
from tqdm import tqdm

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

# ============================================================================
# ГЛОБАЛЬНЫЕ КОНСТАНТЫ
# ============================================================================

# Имена колонок в выходных данных
# Используются во всех функциях для единообразия
COLUMNS = [
    "Time",  # время в секундах от начала теста
    "Voltage",  # напряжение ячейки в вольтах
    "Current",  # ток в амперах (+ заряд, - разряд)
    "Ah",  # кулонометрический интеграл в А·ч
    "Wh",  # накопленная энергия в Вт·ч
    "Power",  # мгновенная мощность в Вт
    "Battery_Temp",  # температура ячейки в °C
    "Chamber_Temp",  # температура термокамеры в °C
    "SoC_percent",  # состояние заряда в процентах
    "SoC_fraction"  # состояние заряда в долях (0-1)
]

# Типы файлов для фильтрации (ключевые слова в именах файлов)
# Используются для выбора метода расчёта SoC
FILE_TYPE_KEYWORDS = {
    "c20_calibration": ["C20", "OCV"],  # калибровочный тест
    "hppc": ["5pulse", "HPPC"],  # импульсный тест HPPC
    "drive_cycle": ["Cycle", "US06", "HWFET", "UDDS", "LA92", "NN"],  # ездовые циклы
    "charge": ["Charge"],  # зарядные тесты
    "pause": ["Pause"],  # паузы
    "discharge": ["dis5_10p", "Dis1C", "Dis_"]  # разрядные тесты
}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def get_file_type(filepath: str) -> str:
    """
    Определяет тип файла по пути и имени.

    Parameters
    ----------
    filepath : str
        Полный путь к файлу или имя файла

    Returns
    -------
    str
        Тип файла: 'c20_calibration', 'hppc', 'drive_cycle', 'charge',
        'pause', 'discharge' или 'other'
    """
    file_lower = os.path.basename(filepath).lower()

    for file_type, keywords in FILE_TYPE_KEYWORDS.items():
        if any(kw.lower() in file_lower for kw in keywords):
            return file_type

    return "other"


def parse_panasonic_mat(file_path: str):
    """
    Загрузка .mat файла Panasonic и преобразование в DataFrame.

    Файлы Panasonic имеют структуру 'meas' с полями:
    - Time: время в секундах
    - Voltage: напряжение в вольтах
    - Current: ток в амперах
    - Ah: кулонометрический интеграл в А·ч
    - и другие

    Parameters
    ----------
    file_path : str
        Путь к .mat файлу

    Returns
    -------
    pd.DataFrame or None
        DataFrame с колонками COLUMNS или None при ошибке
    """
    try:
        # Загружаем .mat файл
        # struct_as_record=False позволяет получить доступ к полям через .fieldname
        # squeeze_me=True убирает лишние размерности
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

        # Извлекаем структуру 'meas' (все файлы Panasonic имеют такую структуру)
        meas = mat['meas']

        # Создаём DataFrame из полей структуры
        df = pd.DataFrame({
            COLUMNS[0]: getattr(meas, 'Time'),  # время
            COLUMNS[1]: getattr(meas, 'Voltage'),  # напряжение
            COLUMNS[2]: getattr(meas, 'Current'),  # ток
            COLUMNS[3]: getattr(meas, 'Ah'),  # кулонометрический интеграл
            COLUMNS[4]: getattr(meas, 'Wh'),  # энергия
            COLUMNS[5]: getattr(meas, 'Power'),  # мощность
            COLUMNS[6]: getattr(meas, 'Battery_Temp_degC'),  # температура ячейки
            COLUMNS[7]: getattr(meas, 'Chamber_Temp_degC'),  # температура камеры
        })

        # ========== РАСЧЁТ ЭТАЛОННОГО SoC ИЗ Ah ==========
        # Поле Ah содержит кулонометрический интеграл.
        # При полном разряде Ah достигает минимума (0% SoC)
        # При полном заряде Ah достигает максимума (100% SoC)
        ah_min = df[COLUMNS[3]].min()
        ah_max = df[COLUMNS[3]].max()
        delta_ah = ah_max - ah_min

        if delta_ah > 0:
            # Линейная интерполяция между min и max даёт SoC
            df[COLUMNS[9]] = (df[COLUMNS[3]] - ah_min) / delta_ah
            df[COLUMNS[8]] = df[COLUMNS[9]] * 100
        else:
            # Если Ah не меняется (маловероятно), ставим 50%
            df[COLUMNS[9]] = 0.5
            df[COLUMNS[8]] = 50.0

        return df
    except Exception as e:
        print(f"Ошибка загрузки {os.path.basename(file_path)}: {e}")
        return None


def build_ocv_soc_interpolation(c20_df: pd.DataFrame):
    """
    Создание функций интерполяции для связи напряжения и SoC из C20 теста.

    C20 тест — это медленный разряд и заряд током C/20.
    При таком малом токе падением напряжения на внутреннем сопротивлении
    можно пренебречь, поэтому Voltage ≈ OCV (напряжение разомкнутой цепи).

    Parameters
    ----------
    c20_df : pd.DataFrame
        DataFrame с данными C20 теста (должен содержать Voltage, Current, SoC_fraction)

    Returns
    -------
    tuple
        (charge_interp, discharge_interp) - функции интерполяции
        charge_interp: напряжение -> SoC для заряда
        discharge_interp: напряжение -> SoC для разряда
    """
    # ========== РАЗРЯДНАЯ ЧАСТЬ ==========
    # Берём только точки с током < 0 (разряд)
    df_discharge = c20_df[c20_df[COLUMNS[2]] < 0].copy()
    discharge_interp = None

    if len(df_discharge) > 0:
        # Сортируем по убыванию напряжения (от 4.2В к 2.5В)
        df_discharge = df_discharge.sort_values(COLUMNS[1], ascending=False)
        # Создаём интерполяционную функцию
        discharge_interp = interp1d(
            df_discharge[COLUMNS[1]],
            df_discharge[COLUMNS[9]],
            bounds_error=False,
            fill_value="extrapolate"  # экстраполяция за пределы диапазона
        )

    # ========== ЗАРЯДНАЯ ЧАСТЬ ==========
    # Берём только точки с током > 0 (заряд)
    df_charge = c20_df[c20_df[COLUMNS[2]] > 0].copy()
    charge_interp = None

    if len(df_charge) > 0:
        # Сортируем по возрастанию напряжения (от 2.5В к 4.2В)
        df_charge = df_charge.sort_values(COLUMNS[1], ascending=True)
        charge_interp = interp1d(
            df_charge[COLUMNS[1]],
            df_charge[COLUMNS[9]],
            bounds_error=False,
            fill_value="extrapolate"
        )

    return charge_interp, discharge_interp


def get_max_capacities_from_c20(c20_df: pd.DataFrame):
    """
    Получение максимальной ёмкости заряда и разряда из C20 теста.

    C20 тест показывает истинную ёмкость ячейки при данном токе.
    Используется для калибровки кулонометрического счёта.

    Parameters
    ----------
    c20_df : pd.DataFrame
        DataFrame с данными C20 теста

    Returns
    -------
    tuple
        (max_charge_capacity, max_discharge_capacity) в А·ч
    """
    # Разрядная ёмкость = |Ah_max - Ah_min| для разрядной части
    df_discharge = c20_df[c20_df[COLUMNS[2]] < 0]
    df_charge = c20_df[c20_df[COLUMNS[2]] > 0]

    # Значения по умолчанию (паспортная ёмкость)
    max_discharge = 2.9
    max_charge = 2.9

    if len(df_discharge) > 0:
        max_discharge = abs(df_discharge[COLUMNS[3]].max() - df_discharge[COLUMNS[3]].min())

    if len(df_charge) > 0:
        max_charge = abs(df_charge[COLUMNS[3]].max() - df_charge[COLUMNS[3]].min())

    return max_charge, max_discharge


def find_c20_calibration_file(parsed_dir: str) -> str | None:
    """
    Находит C20 калибровочный файл ТОЛЬКО в папке 25degC.

    Согласно документации, C20 тест для калибровки OCV(SOC)
    выполняется ТОЛЬКО при комнатной температуре (+25°C).

    Parameters
    ----------
    parsed_dir : str
        Путь к папке с parsed CSV файлами

    Returns
    -------
    str or None
        Путь к C20 файлу или None
    """
    # Стандартный путь по документации
    search_path = os.path.join(parsed_dir, "25degC", "C20 OCV and 1C discharge tests_start_of_tests")

    if not os.path.exists(search_path):
        # Поиск в подпапках 25degC
        for root, dirs, files in os.walk(os.path.join(parsed_dir, "25degC")):
            for file in files:
                if 'C20' in file and 'OCV' in file and file.endswith('_parsed.csv'):
                    return os.path.join(root, file)
        return None

    # Проверка известных имён файлов (из анализа датасета)
    candidates = [
        "05-08-17_13.26 C20 OCV Test_C20_25dC_parsed.csv",
        "C20 OCV Test_C20_25dC_parsed.csv"
    ]

    for candidate in candidates:
        full_path = os.path.join(search_path, candidate)
        if os.path.exists(full_path):
            return full_path

    # Любой CSV в этой папке
    for file in os.listdir(search_path):
        if file.endswith('_parsed.csv'):
            return os.path.join(search_path, file)

    return None


def get_initial_soc_by_voltage(df: pd.DataFrame, charge_interp, discharge_interp) -> float:
    """
    Определение начального состояния заряда по напряжению.

    Использует интерполяционные функции из C20 теста.
    По первому напряжению и знаку первого ненулевого тока
    определяет начальное SoC.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с данными
    charge_interp : callable
        Функция интерполяции для заряда (напряжение -> SoC)
    discharge_interp : callable
        Функция интерполяции для разряда (напряжение -> SoC)

    Returns
    -------
    float
        Начальное SoC в долях (0-1)
    """
    if len(df) == 0:
        return 0.5

    # Берём первое напряжение
    initial_voltage = df[COLUMNS[1]].iloc[0]

    # Находим первый ненулевой ток, чтобы понять режим (заряд/разряд)
    non_zero_current = df[df[COLUMNS[2]] != 0]

    if len(non_zero_current) == 0:
        # Если ток всё время нулевой (чистая пауза)
        if discharge_interp is not None:
            return np.clip(discharge_interp(initial_voltage), 0, 1)
        return 0.5

    first_current = non_zero_current[COLUMNS[2]].iloc[0]

    if first_current < 0:  # Разряд
        if discharge_interp is not None:
            return np.clip(discharge_interp(initial_voltage), 0, 1)
    else:  # Заряд
        if charge_interp is not None:
            return np.clip(charge_interp(initial_voltage), 0, 1)
        if discharge_interp is not None:
            return np.clip(discharge_interp(initial_voltage), 0, 1)

    return 0.5


def calculate_soc_coulomb(df: pd.DataFrame, initial_soc: float,
                          max_charge_cap: float, max_discharge_cap: float) -> np.ndarray:
    """
    Кулонометрический расчёт SoC.

    SoC изменяется пропорционально интегралу тока.
    Для разряда: SoC = SoC_initial - |∫I dt| / Q_discharge
    Для заряда:   SoC = SoC_initial + ∫I dt / Q_charge

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame с колонками Time и Current
    initial_soc : float
        Начальное SoC в долях (0-1)
    max_charge_cap : float
        Максимальная зарядная ёмкость в А·ч
    max_discharge_cap : float
        Максимальная разрядная ёмкость в А·ч

    Returns
    -------
    np.ndarray
        Массив рассчитанных значений SoC (0-1)
    """
    if len(df) == 0:
        return np.array([])

    # Расчёт времени в часах (дифференциал времени)
    time_diff = df[COLUMNS[0]].diff().fillna(0) / 3600
    # Изменение ёмкости = ток * время
    delta_ah = df[COLUMNS[2]] * time_diff
    # Накопленная ёмкость
    cumulative_ah = delta_ah.cumsum()

    soc_values = []
    for current, cum_ah in zip(df[COLUMNS[2]], cumulative_ah):
        if current < 0:  # Разряд
            soc = initial_soc - (abs(cum_ah) / max_discharge_cap)
        else:  # Заряд или пауза
            soc = initial_soc + (cum_ah / max_charge_cap)

        # Клиппинг до [0, 1]
        soc_values.append(np.clip(soc, 0, 1))

    return np.array(soc_values)


def check_directories_exist(data_directory_dict):
    """
    Проверка существования директорий.

    Parameters
    ----------
    data_directory_dict : dict
        Словарь с путями к директориям

    Returns
    -------
    tuple
        (existing_non_empty_dirs, missing_or_empty_dirs)
    """
    existing_non_empty_dirs = []
    missing_or_empty_dirs = []

    for name, path in data_directory_dict.items():
        if os.path.exists(path) and os.path.isdir(path):
            if any(os.listdir(path)):
                existing_non_empty_dirs.append(name)
            else:
                missing_or_empty_dirs.append((name, "directory_empty"))
        else:
            missing_or_empty_dirs.append((name, "not_exists"))

    return existing_non_empty_dirs, missing_or_empty_dirs


# ============================================================================
# ПЕРВЫЙ ЭТАП: КОНВЕРТАЦИЯ .mat В parsed CSV
# ============================================================================

def files_processor_first(args, data_and_parsed_directory_dict):
    """
    Конвертация .mat файла в parsed CSV.
    Сохраняет структуру папок (температура/подпапка).

    Parameters
    ----------
    args : tuple
        (mat_file_name, temperature, subfolder)
    data_and_parsed_directory_dict : dict
        Словарь с путями к директориям данных
    """
    mat_file_name, temperature, subfolder = args

    # Формируем путь для сохранения parsed файла
    # Сохраняем структуру: parsed/температура/подпапка/
    parsed_dir = os.path.join(data_and_parsed_directory_dict["Panasonic_parsed"], temperature)
    if subfolder:
        parsed_dir = os.path.join(parsed_dir, subfolder)

    os.makedirs(parsed_dir, exist_ok=True)

    # Путь к исходному .mat файлу
    crude_file_path = os.path.join(
        data_and_parsed_directory_dict["Panasonic_data"],
        temperature,
        subfolder if subfolder else "",
        f"{mat_file_name}.mat"
    )

    if not os.path.exists(crude_file_path):
        return

    # Парсинг .mat файла
    df = parse_panasonic_mat(crude_file_path)

    if df is None:
        return

    # Пропускаем файлы с нулевым током (чистые паузы без данных)
    # Такие файлы неинформативны для обучения
    if (df[COLUMNS[2]] == 0).all():
        return

    # Добавляем информацию о типе файла для использования на втором этапе
    df['File_Type'] = get_file_type(crude_file_path)

    # Сохраняем parsed CSV
    parsed_file_path = os.path.join(parsed_dir, f"{mat_file_name}_parsed.csv")
    df.to_csv(parsed_file_path, index=False)


# ============================================================================
# ВТОРОЙ ЭТАП: ОБРАБОТКА parsed В processed (РАСЧЁТ SoC)
# ============================================================================

def files_processor_second(args, data_and_parsed_directory_dict, calibration_params):
    """
    Обработка parsed файлов с использованием калибровки от C20.
    Рассчитывает SoC кулонометрическим методом для разных типов файлов.

    Для разных типов файлов используются разные методы расчёта SoC:
    - C20: прямой расчёт из Ah
    - HPPC: эталонный SoC из Ah
    - Drive cycles, charge, pause: кулонометрический метод с калибровкой от C20

    Parameters
    ----------
    args : tuple
        (mat_file_name, temperature, subfolder)
    data_and_parsed_directory_dict : dict
        Словарь с путями к директориям данных
    calibration_params : dict
        Параметры калибровки из C20 теста
    """
    mat_file_name, temperature, subfolder = args

    # Формирование путей
    parsed_dir = os.path.join(data_and_parsed_directory_dict["Panasonic_parsed"], temperature)
    processed_dir = os.path.join(data_and_parsed_directory_dict["Panasonic_processed"], temperature)

    if subfolder:
        parsed_dir = os.path.join(parsed_dir, subfolder)
        processed_dir = os.path.join(processed_dir, subfolder)

    parsed_file_path = os.path.join(parsed_dir, f"{mat_file_name}_parsed.csv")

    if not os.path.exists(parsed_file_path):
        return

    # Загрузка parsed данных
    df = pd.read_csv(parsed_file_path)

    # Восстановление числовых типов (при сохранении в CSV они могли стать строками)
    for col in COLUMNS[:8]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Определение типа файла
    file_type = df['File_Type'].iloc[0] if 'File_Type' in df.columns else 'other'

    # Получение параметров калибровки
    charge_interp = calibration_params.get('charge_interp')
    discharge_interp = calibration_params.get('discharge_interp')
    max_charge_cap = calibration_params.get('max_charge_capacity', 2.9)
    max_discharge_cap = calibration_params.get('max_discharge_capacity', 2.9)

    # ========== ВЫБОР МЕТОДА РАСЧЁТА SoC В ЗАВИСИМОСТИ ОТ ТИПА ФАЙЛА ==========

    if file_type == 'c20_calibration':
        # Для C20 файлов используем прямой расчёт из Ah
        # Это эталонный SoC
        if COLUMNS[3] in df.columns:
            ah_min = df[COLUMNS[3]].min()
            ah_max = df[COLUMNS[3]].max()
            if ah_max - ah_min > 0:
                df[COLUMNS[9]] = (df[COLUMNS[3]] - ah_min) / (ah_max - ah_min)
                df[COLUMNS[8]] = df[COLUMNS[9]] * 100

    elif file_type == 'hppc':
        # Для HPPC файлов тоже используем прямой расчёт из Ah
        # HPPC тесты содержат паузы для OCV-калибровки, Ah уже откалиброван
        if COLUMNS[3] in df.columns:
            ah_min = df[COLUMNS[3]].min()
            ah_max = df[COLUMNS[3]].max()
            if ah_max - ah_min > 0:
                df[COLUMNS[9]] = (df[COLUMNS[3]] - ah_min) / (ah_max - ah_min)
                df[COLUMNS[8]] = df[COLUMNS[9]] * 100
            else:
                # Fallback на кулонометрический метод
                initial_soc = get_initial_soc_by_voltage(df, charge_interp, discharge_interp)
                soc = calculate_soc_coulomb(df, initial_soc, max_charge_cap, max_discharge_cap)
                df[COLUMNS[9]] = soc
                df[COLUMNS[8]] = soc * 100

    else:
        # Для ездовых циклов, зарядов и пауз используем кулонометрический метод
        # с калибровкой от C20 теста
        if discharge_interp is not None:
            initial_soc = get_initial_soc_by_voltage(df, charge_interp, discharge_interp)
            soc = calculate_soc_coulomb(df, initial_soc, max_charge_cap, max_discharge_cap)
            df[COLUMNS[9]] = soc
            df[COLUMNS[8]] = soc * 100
        elif COLUMNS[3] in df.columns:
            # Fallback: прямой расчёт из Ah
            ah_min = df[COLUMNS[3]].min()
            ah_max = df[COLUMNS[3]].max()
            if ah_max - ah_min > 0:
                df[COLUMNS[9]] = (df[COLUMNS[3]] - ah_min) / (ah_max - ah_min)
                df[COLUMNS[8]] = df[COLUMNS[9]] * 100
            else:
                # Нет данных для расчёта
                return

    # Округление времени для уменьшения размера данных (опционально)
    # Удаляем дубликаты по округлённому времени
    if COLUMNS[0] in df.columns:
        df['Rounded_Time'] = df[COLUMNS[0]].round().astype(int)
        df_processed = df.drop_duplicates(subset='Rounded_Time')
    else:
        df_processed = df

    # Сохранение processed CSV
    os.makedirs(processed_dir, exist_ok=True)
    processed_file_path = os.path.join(processed_dir, f'{mat_file_name}_processed.csv')
    df_processed.to_csv(processed_file_path, index=False)


# ============================================================================
# СБОР .mat ФАЙЛОВ
# ============================================================================

def collect_mat_files(data_dir: str):
    """
    Рекурсивный сбор всех .mat файлов с сохранением структуры папок.

    Parameters
    ----------
    data_dir : str
        Путь к корневой папке с данными Panasonic

    Returns
    -------
    list
        Список кортежей (file_name, temperature, subfolder)
        temperature: температура (25degC, 10degC, 0degC, -10degC, -20degC)
        subfolder: подпапка (например, "Drive cycles", "5 pulse disch" и т.д.)
    """
    tasks_args = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                # Относительный путь от корня датасета
                rel_path = os.path.relpath(root, data_dir)
                file_name = file.replace('.mat', '')

                # Разбор пути: первая часть — температура, остальное — подпапки
                path_parts = rel_path.split(os.sep)
                temperature = path_parts[0] if path_parts and path_parts[0] != '.' else ""
                subfolder = os.sep.join(path_parts[1:]) if len(path_parts) > 1 else ""

                tasks_args.append((file_name, temperature, subfolder))

    return tasks_args


# ============================================================================
# ЗАГРУЗКА C20 КАЛИБРОВКИ
# ============================================================================

def load_c20_calibration(parsed_dir: str) -> dict:
    """
    Загружает C20 калибровочный файл из папки 25degC.

    Согласно документации, C20 тест для калибровки OCV(SOC)
    выполняется ТОЛЬКО при комнатной температуре (+25°C).

    Parameters
    ----------
    parsed_dir : str
        Путь к папке с parsed CSV файлами

    Returns
    -------
    dict
        Словарь с параметрами калибровки:
        - charge_interp: функция интерполяции для заряда
        - discharge_interp: функция интерполяции для разряда
        - max_charge_capacity: максимальная зарядная ёмкость
        - max_discharge_capacity: максимальная разрядная ёмкость
        - c20_loaded: флаг успешной загрузки
    """
    c20_file = find_c20_calibration_file(parsed_dir)

    if not c20_file:
        print("\n❌ ВНИМАНИЕ: C20 калибровочный файл не найден в папке 25degC!")
        print("   Будут использованы значения по умолчанию (ёмкость 2.9 А·ч)")
        print("   Возможные причины: данные не загружены или структура папок изменена")
        return {
            'charge_interp': None,
            'discharge_interp': None,
            'max_charge_capacity': 2.9,
            'max_discharge_capacity': 2.9,
            'c20_loaded': False
        }

    print(f"\n✅ Загружен C20 файл: {os.path.basename(c20_file)}")

    # Загрузка и подготовка данных
    c20_df = pd.read_csv(c20_file)
    for col in COLUMNS[:8]:
        if col in c20_df.columns:
            c20_df[col] = pd.to_numeric(c20_df[col], errors='coerce')

    # Создание интерполяционных функций OCV -> SoC
    charge_interp, discharge_interp = build_ocv_soc_interpolation(c20_df)

    # Расчёт максимальных ёмкостей из C20 теста
    max_charge_cap, max_discharge_cap = get_max_capacities_from_c20(c20_df)

    print(f"   Макс. зарядная ёмкость:   {max_charge_cap:.4f} А·ч")
    print(f"   Макс. разрядная ёмкость: {max_discharge_cap:.4f} А·ч")

    return {
        'charge_interp': charge_interp,
        'discharge_interp': discharge_interp,
        'max_charge_capacity': max_charge_cap,
        'max_discharge_capacity': max_discharge_cap,
        'c20_loaded': True
    }


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main(data_directory_dict):
    """
    Главная функция обработки датасета Panasonic.

    Выполняет два этапа обработки:
    1. Конвертация .mat -> parsed CSV (сохранение всех исходных данных)
    2. Расчёт SoC и сохранение processed CSV (с добавлением колонки SoC)

    Parameters
    ----------
    data_directory_dict : dict
        Словарь с путями к директориям:
        - Panasonic_data: исходные .mat файлы
        - Panasonic_parsed: промежуточные CSV
        - Panasonic_processed: финальные CSV с SoC
        - Panasonic_parsed_plots: (опционально) для графиков
        - Panasonic_processed_plots: (опционально) для графиков
    """

    # ========== ПРОВЕРКА ДИРЕКТОРИЙ ==========
    existing, missing = check_directories_exist(data_directory_dict)
    print(f"Существующие директории: {existing}")

    # Если все директории существуют и не пусты — не запускаемся
    # Это защита от случайного повторного запуска
    if not missing:
        print("Все директории существуют. Программа не запускается.")
        print("Если нужно переобработать данные, удалите папки Panasonic_parsed и Panasonic_processed")
        return

    print(f"Обнаружены пустые/отсутствующие директории: {missing}")

    # Создание недостающих директорий
    for dir_name, dir_path in list(data_directory_dict.items())[1:]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Создана директория: {dir_path}")

    # ========== СБОР .mat ФАЙЛОВ ==========
    print("\n📂 Поиск .mat файлов...")
    tasks_args = collect_mat_files(data_directory_dict["Panasonic_data"])
    print(f"Найдено {len(tasks_args)} .mat файлов для обработки")

    if len(tasks_args) == 0:
        print("❌ .mat файлы не найдены!")
        print(f"   Проверьте путь: {data_directory_dict['Panasonic_data']}")
        return

    # Количество процессов для параллельной обработки
    num_processes = 4
    print(f"Используется процессов: {num_processes}")

    # ========== ЭТАП 1: Конвертация .mat в parsed CSV ==========
    print("\n=== Этап 1: Конвертация .mat в parsed CSV ===")
    print("   Результат: файлы *_parsed.csv с原始ными данными")

    with Pool(num_processes) as pool:
        tasks_with_dict = [(task, data_directory_dict) for task in tasks_args]
        list(tqdm(
            pool.starmap(files_processor_first, tasks_with_dict),
            total=len(tasks_args),
            desc="Обработка .mat файлов"
        ))

    print("✅ Этап 1 завершён")

    # ========== ЗАГРУЗКА КАЛИБРОВКИ ИЗ C20 ==========
    print("\n=== Загрузка калибровки из C20 теста (только 25°C) ===")
    print("   C20 тест используется для:\n"
          "   1. Создания интерполяционных функций OCV -> SoC\n"
          "   2. Определения максимальной ёмкости ячейки")
    calibration_params = load_c20_calibration(data_directory_dict["Panasonic_parsed"])

    # ========== ЭТАП 2: Обработка parsed в processed (расчёт SoC) ==========
    print("\n=== Этап 2: Расчёт SoC и сохранение processed CSV ===")
    print("   Для разных типов файлов используются разные методы расчёта:\n"
          "   - C20 и HPPC: прямой расчёт из Ah (эталонный SoC)\n"
          "   - Drive cycles, charge, pause: кулонометрический метод с калибровкой от C20")

    with Pool(num_processes) as pool:
        tasks_with_params = [(task, data_directory_dict, calibration_params) for task in tasks_args]
        list(tqdm(
            pool.starmap(files_processor_second, tasks_with_params),
            total=len(tasks_args),
            desc="Постобработка файлов"
        ))

    print("✅ Этап 2 завершён")

    # ========== ИТОГИ ==========
    print("\n" + "=" * 80)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 80)

    # Подсчёт обработанных файлов
    processed_count = 0
    for root, dirs, files in os.walk(data_directory_dict["Panasonic_processed"]):
        processed_count += len([f for f in files if f.endswith('_processed.csv')])

    print(f"\n📊 Статистика:")
    print(f"   - Исходных .mat файлов: {len(tasks_args)}")
    print(f"   - Успешно обработано: {processed_count}")
    print(f"   - Калибровка из C20: {'ДА' if calibration_params['c20_loaded'] else 'НЕТ'}")

    if calibration_params['c20_loaded']:
        print(f"   - Ёмкость из C20: {calibration_params['max_discharge_capacity']:.3f} А·ч")
    else:
        print(f"   - Использована паспортная ёмкость: 2.9 А·ч")

    print(f"\n📁 Результаты сохранены в:")
    print(f"   - Parsed данные: {data_directory_dict['Panasonic_parsed']}")
    print(f"   - Processed данные: {data_directory_dict['Panasonic_processed']}")
    print(f"\n   Структура папок сохранена: температура/подпапка/")
    print("\n" + "=" * 80)


# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == '__main__':
    # ========================================================================
    # НАСТРОЙКА ПУТЕЙ (ИЗМЕНИТЕ ПОД ВАШУ СТРУКТУРУ)
    # ========================================================================

    # Основная директория с данными Panasonic
    # Должна содержать папку "Panasonic 18650PF Data" с .mat файлами
    MAIN_DIRECTORY = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2"

    # Словарь с путями
    # Структура аналогичная вашему коду
    data_directory_dict = {
        "Panasonic_data": f"{MAIN_DIRECTORY}/Panasonic 18650PF Data",  # исходные .mat файлы
        "Panasonic_parsed": f"{MAIN_DIRECTORY}/Panasonic_parsed",  # промежуточные CSV
        "Panasonic_processed": f"{MAIN_DIRECTORY}/Panasonic_processed",  # финальные CSV с SoC
        "Panasonic_parsed_plots": f"{MAIN_DIRECTORY}/Panasonic_parsed_plots",  # для графиков
        "Panasonic_processed_plots": f"{MAIN_DIRECTORY}/Panasonic_processed_plots"  # для графиков
    }

    # Запуск обработки
    main(data_directory_dict)