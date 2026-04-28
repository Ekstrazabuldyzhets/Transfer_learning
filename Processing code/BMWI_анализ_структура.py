import os
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 1"
BMWI_SOURCE = os.path.join(SOURCE_PATH, "Measurement Data")

CSV_SEP = ';'
CSV_ENCODING = 'latin1'

# Ограничение для вывода (чтобы не перегружать консоль)
MAX_FILES_DETAIL = 70  # Будем выводить все файлы детально
MAX_COLS_SHOW = 10


# ============================================
# ФУНКЦИЯ ЗАГРУЗКИ И ПРЕОБРАЗОВАНИЯ
# ============================================

def load_and_convert(filepath):
    """
    Загружает CSV файл BMW i3 и преобразует числовые колонки
    """
    try:
        # Читаем как строки, чтобы потом преобразовать запятые в точки
        df_raw = pd.read_csv(filepath, sep=CSV_SEP, encoding=CSV_ENCODING, dtype=str)
        df = df_raw.copy()

        # Преобразуем все колонки в числа (убираем запятые)
        for col in df.columns:
            try:
                # Заменяем запятую на точку и конвертируем в число
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
            except:
                pass

        return df
    except Exception as e:
        return None


# ============================================
# ФУНКЦИЯ ПОЛНОГО АНАЛИЗА ОДНОГО ФАЙЛА
# ============================================

def analyze_file(filepath, filename):
    """
    Полный анализ одного CSV файла
    """
    result = {
        'filename': filename,
        'full_path': filepath,
        'size_kb': os.path.getsize(filepath) / 1024,
        'num_rows': 0,
        'num_cols': 0,
        'columns': [],
        'has_nan': False,
        'nan_stats': {},
        'category': 'A' if 'TripA' in filename else ('B' if 'TripB' in filename else 'unknown'),
        'error': None,
        'stats': {}
    }

    df = load_and_convert(filepath)

    if df is None:
        result['error'] = "Не удалось загрузить файл"
        return result

    result['num_rows'] = len(df)
    result['num_cols'] = len(df.columns)
    result['columns'] = list(df.columns)

    # Анализируем каждую колонку
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': df[col].isnull().sum(),
            'null_pct': 100 * df[col].isnull().sum() / len(df) if len(df) > 0 else 0
        }

        if col_info['null_count'] > 0:
            result['has_nan'] = True
            result['nan_stats'][col] = col_info['null_count']

        # Статистика для числовых колонок
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                col_info['min'] = float(clean_data.min())
                col_info['max'] = float(clean_data.max())
                col_info['mean'] = float(clean_data.mean())
                col_info['std'] = float(clean_data.std())

        result['stats'][col] = col_info

    # Дополнительная информация о поездке
    if 'Time [s]' in df.columns and pd.api.types.is_numeric_dtype(df['Time [s]']):
        time_clean = df['Time [s]'].dropna()
        if len(time_clean) > 0:
            result['duration_s'] = time_clean.max()
            result['duration_min'] = time_clean.max() / 60
            # Частота дискретизации
            dt = time_clean.diff().median()
            result['fs'] = 1 / dt if dt > 0 else np.nan

    if 'SoC [%]' in df.columns and pd.api.types.is_numeric_dtype(df['SoC [%]']):
        soc_clean = df['SoC [%]'].dropna()
        if len(soc_clean) > 0:
            result['soc_start'] = float(soc_clean.iloc[0])
            result['soc_end'] = float(soc_clean.iloc[-1])
            result['soc_consumed'] = result['soc_start'] - result['soc_end']

    # Информация о нагревателе (есть только в категории B)
    if 'Heating Power CAN [kW]' in df.columns and pd.api.types.is_numeric_dtype(df['Heating Power CAN [kW]']):
        heater = df['Heating Power CAN [kW]'].dropna()
        if len(heater) > 0:
            result['heater_max'] = float(heater.max())
            result['heater_mean'] = float(heater.mean())
            result['heater_time_on_min'] = (heater > 0.1).sum() / 10 / 60

    return result


# ============================================
# ФУНКЦИЯ ВЫВОДА ИНФОРМАЦИИ О ФАЙЛЕ (ПОЛНЫЙ)
# ============================================

def print_file_analysis_full(result, idx, total):
    """
    Выводит полный анализ одного файла
    """
    print(f"\n{'=' * 120}")
    print(f"[{idx}/{total}] 📄 {result['filename']}")
    print(f"{'=' * 120}")
    print(f"   Размер: {result['size_kb']:.1f} KB")
    print(f"   Категория: {result['category']}")

    if result['error']:
        print(f"   ❌ ОШИБКА: {result['error']}")
        return

    print(f"   ✅ УСПЕШНО ЗАГРУЖЕНО")
    print(f"   Строк: {result['num_rows']:,}")
    print(f"   Столбцов: {result['num_cols']}")

    # Показываем список колонок
    print(f"\n   📋 СПИСОК КОЛОНОК:")
    for i, col in enumerate(result['columns']):
        marker = " ⚠️" if result['stats'][col]['null_count'] > 0 else ""
        print(f"      [{i:2d}] {col}{marker}")
        if i >= 19 and len(result['columns']) > 20:
            print(f"      ... и еще {len(result['columns']) - 20} колонок")
            break

    print(f"\n   📊 СТАТИСТИКА ПО КЛЮЧЕВЫМ КОЛОНКАМ:")

    # Важные колонки для анализа
    key_cols = [
        'Time [s]',
        'Velocity [km/h]',
        'Battery Voltage [V]',
        'Battery Current [A]',
        'SoC [%]',
        'Battery Temperature [°C]',
        'Ambient Temperature [°C]',
        'Heating Power CAN [kW]'
    ]

    for col in key_cols:
        if col in result['stats']:
            stats = result['stats'][col]
            print(f"\n   🔹 {col}:")
            print(f"      тип: {stats['dtype']}")
            if stats['null_count'] > 0:
                print(f"      пропуски: {stats['null_count']:,} ({stats['null_pct']:.2f}%)")
            if 'min' in stats:
                print(f"      min: {stats['min']:.4f}")
                print(f"      max: {stats['max']:.4f}")
                print(f"      mean: {stats['mean']:.4f}")
                print(f"      std: {stats['std']:.4f}")

    # Дополнительная информация о поездке
    print(f"\n   🚗 ИНФОРМАЦИЯ О ПОЕЗДКЕ:")
    if 'duration_min' in result:
        print(f"      Длительность: {result['duration_min']:.1f} мин ({result['duration_s'] / 3600:.2f} ч)")
    if 'fs' in result:
        print(f"      Частота дискретизации: {result['fs']:.2f} Гц")
    if 'soc_start' in result:
        print(f"      SoC начало: {result['soc_start']:.1f}%")
        print(f"      SoC конец: {result['soc_end']:.1f}%")
        print(f"      Расход SoC: {result['soc_consumed']:.1f}%")

    if 'heater_max' in result:
        print(f"\n   🔥 PTC-НАГРЕВАТЕЛЬ:")
        print(f"      Макс. мощность: {result['heater_max']:.2f} кВт")
        print(f"      Ср. мощность: {result['heater_mean']:.2f} кВт")
        print(f"      Время работы: {result['heater_time_on_min']:.1f} мин")

    if result['has_nan']:
        print(f"\n   ⚠️ ВНИМАНИЕ: есть пропуски в колонках:")
        for col, count in list(result['nan_stats'].items())[:5]:
            print(f"      - {col}: {count} пропусков")
        if len(result['nan_stats']) > 5:
            print(f"      ... и еще {len(result['nan_stats']) - 5} колонок")


# ============================================
# ФУНКЦИЯ СБОРА ВСЕХ ФАЙЛОВ
# ============================================

def collect_files(base_path):
    """
    Собирает все CSV файлы
    """
    files = []
    for f in os.listdir(base_path):
        if f.endswith('.csv'):
            full_path = os.path.join(base_path, f)
            files.append((full_path, f))
    return files


# ============================================
# ФУНКЦИЯ СВОДНОЙ СТАТИСТИКИ ПО КАТЕГОРИЯМ
# ============================================

def print_category_summary(results, category_name):
    """
    Печатает сводку по категории
    """
    cat_results = [r for r in results if r['category'] == category_name and not r['error']]

    if not cat_results:
        print(f"\nНет данных для категории {category_name}")
        return

    print(f"\n{'=' * 120}")
    print(f"📊 СВОДНАЯ СТАТИСТИКА ПО КАТЕГОРИИ {category_name}")
    print(f"{'=' * 120}")
    print(f"   Файлов: {len(cat_results)}")
    print(f"   Всего строк: {sum(r['num_rows'] for r in cat_results):,}")

    # Собираем все значения параметров
    all_vel_min = [r['stats']['Velocity [km/h]']['min'] for r in cat_results if 'Velocity [km/h]' in r['stats']]
    all_vel_max = [r['stats']['Velocity [km/h]']['max'] for r in cat_results if 'Velocity [km/h]' in r['stats']]
    all_voltage_min = [r['stats']['Battery Voltage [V]']['min'] for r in cat_results if
                       'Battery Voltage [V]' in r['stats']]
    all_voltage_max = [r['stats']['Battery Voltage [V]']['max'] for r in cat_results if
                       'Battery Voltage [V]' in r['stats']]
    all_current_min = [r['stats']['Battery Current [A]']['min'] for r in cat_results if
                       'Battery Current [A]' in r['stats']]
    all_current_max = [r['stats']['Battery Current [A]']['max'] for r in cat_results if
                       'Battery Current [A]' in r['stats']]
    all_soc_min = [r['stats']['SoC [%]']['min'] for r in cat_results if 'SoC [%]' in r['stats']]
    all_soc_max = [r['stats']['SoC [%]']['max'] for r in cat_results if 'SoC [%]' in r['stats']]
    all_temp_min = [r['stats']['Battery Temperature [°C]']['min'] for r in cat_results if
                    'Battery Temperature [°C]' in r['stats']]
    all_temp_max = [r['stats']['Battery Temperature [°C]']['max'] for r in cat_results if
                    'Battery Temperature [°C]' in r['stats']]
    all_durations = [r['duration_min'] for r in cat_results if 'duration_min' in r]
    all_soc_consumed = [r['soc_consumed'] for r in cat_results if 'soc_consumed' in r]

    print(f"\n   📈 ДИАПАЗОНЫ ПАРАМЕТРОВ:")
    if all_vel_min:
        print(f"      Скорость: {min(all_vel_min):.0f} – {max(all_vel_max):.0f} км/ч")
    if all_voltage_min:
        print(f"      Напряжение: {min(all_voltage_min):.0f} – {max(all_voltage_max):.0f} В")
    if all_current_min:
        print(f"      Ток: {min(all_current_min):.0f} – {max(all_current_max):.0f} А")
    if all_soc_min:
        print(f"      SoC: {min(all_soc_min):.0f} – {max(all_soc_max):.0f} %")
    if all_temp_min:
        print(f"      Температура батареи: {min(all_temp_min):.1f} – {max(all_temp_max):.1f} °C")
    if all_durations:
        print(f"      Длительность: {min(all_durations):.1f} – {max(all_durations):.1f} мин")
    if all_soc_consumed:
        print(f"      Расход SoC: {min(all_soc_consumed):.1f} – {max(all_soc_consumed):.1f} %")

    # Статистика по нагревателю (только для категории B)
    if category_name == 'B':
        all_heater_max = [r['heater_max'] for r in cat_results if 'heater_max' in r]
        all_heater_mean = [r['heater_mean'] for r in cat_results if 'heater_mean' in r]
        all_heater_time = [r['heater_time_on_min'] for r in cat_results if 'heater_time_on_min' in r]

        if all_heater_max:
            print(f"\n   🔥 PTC-НАГРЕВАТЕЛЬ:")
            print(f"      Макс. мощность: {max(all_heater_max):.2f} кВт")
            print(f"      Ср. мощность: {np.mean(all_heater_mean):.2f} кВт")
            print(f"      Время работы: {np.mean(all_heater_time):.1f} мин")


# ============================================
# ФУНКЦИЯ ДЕТАЛЬНОГО АНАЛИЗА ПРИМЕРНОГО ФАЙЛА
# ============================================

def analyze_example_file(base_path):
    """
    Детальный анализ первого файла категории B
    """
    print("\n" + "=" * 120)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА (категория B)")
    print("=" * 120)

    # Ищем первый файл категории B
    example_file = None
    for f in os.listdir(base_path):
        if 'TripB' in f and f.endswith('.csv'):
            example_file = os.path.join(base_path, f)
            filename = f
            break

    if not example_file:
        print("❌ Не найден файл категории B")
        return

    print(f"\n📄 {filename}")
    print(f"Размер: {os.path.getsize(example_file) / 1024:.1f} KB")

    df = load_and_convert(example_file)
    if df is None:
        print("❌ Не удалось загрузить")
        return

    print(f"\n✅ УСПЕШНО ЗАГРУЖЕНО")
    print(f"📊 Форма: {df.shape[0]} строк × {df.shape[1]} столбцов")

    print(f"\n📋 ВСЕ НАЗВАНИЯ СТОЛБЦОВ:")
    for i, col in enumerate(df.columns):
        print(f"   [{i:2d}] {col}")

    print(f"\n🔍 ПЕРВЫЕ 5 СТРОК (ключевые колонки):")
    key_cols = ['Time [s]', 'Velocity [km/h]', 'Battery Voltage [V]', 'Battery Current [A]', 'SoC [%]',
                'Battery Temperature [°C]']
    available_cols = [c for c in key_cols if c in df.columns]
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df[available_cols].head())

    print(f"\n📈 ПОЛНАЯ СТАТИСТИКА ПО ВСЕМ ЧИСЛОВЫМ СТОЛБЦАМ:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())

    print(f"\n📊 КАЧЕСТВО ДАННЫХ (пропуски):")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"   {col}: {null_count} пропусков ({100 * null_count / len(df):.2f}%)")

    # Информация о частоте дискретизации
    if 'Time [s]' in df.columns and pd.api.types.is_numeric_dtype(df['Time [s]']):
        time_clean = df['Time [s]'].dropna()
        if len(time_clean) > 1:
            dt = time_clean.diff().median()
            print(f"\n🎯 ЧАСТОТА ДИСКРЕТИЗАЦИИ:")
            print(f"   dt = {dt:.4f} с")
            print(f"   Fs = {1 / dt:.2f} Гц")

    # Информация о нагревателе
    if 'Heating Power CAN [kW]' in df.columns:
        heater = df['Heating Power CAN [kW]'].dropna()
        if len(heater) > 0:
            print(f"\n🔥 PTC-НАГРЕВАТЕЛЬ:")
            print(f"   Макс. мощность: {heater.max():.2f} кВт")
            print(f"   Средняя мощность: {heater.mean():.2f} кВт")
            print(f"   Время работы: {(heater > 0.1).sum() / 10 / 60:.1f} мин")


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main():
    if not os.path.exists(BMWI_SOURCE):
        print(f"❌ Путь не найден: {BMWI_SOURCE}")
        return

    print("=" * 120)
    print("🔬 ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА BMW i3 (ПОКАЗАНИЯ ПО КАЖДОМУ ФАЙЛУ)")
    print("=" * 120)
    print(f"\n📁 Путь: {BMWI_SOURCE}")

    # Собираем все файлы
    files = collect_files(BMWI_SOURCE)
    print(f"\n📂 Найдено CSV файлов: {len(files)}")

    # Анализируем каждый файл
    results = []

    for idx, (full_path, filename) in enumerate(files, 1):
        result = analyze_file(full_path, filename)
        results.append(result)

        # Выводим детальную информацию по каждому файлу
        print_file_analysis_full(result, idx, len(files))

    # Сводная статистика по категориям
    print_category_summary(results, 'A')
    print_category_summary(results, 'B')

    # Общая сводка
    print("\n" + "=" * 120)
    print("📊 ИТОГОВАЯ СВОДКА ПО ВСЕМУ ДАТАСЕТУ")
    print("=" * 120)

    successful = [r for r in results if not r['error']]
    failed = [r for r in results if r['error']]

    print(f"\n📁 Всего CSV файлов: {len(files)}")
    print(f"✅ Успешно загружено: {len(successful)}")
    print(f"❌ С ошибками: {len(failed)}")
    print(f"📏 Категория A: {len([r for r in successful if r['category'] == 'A'])} файлов")
    print(f"📏 Категория B: {len([r for r in successful if r['category'] == 'B'])} файлов")

    total_rows = sum(r['num_rows'] for r in successful)
    print(f"📏 Всего строк данных: {total_rows:,}")

    # Детальный анализ примера
    analyze_example_file(BMWI_SOURCE)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ДАТАСЕТА BMW i3 ЗАВЕРШЁН")
    print(f"   Всего проанализировано файлов: {len(results)}")
    print(f"   Успешно: {len(successful)}")
    print(f"   С ошибками: {len(failed)}")
    print("=" * 120)


if __name__ == "__main__":
    main()