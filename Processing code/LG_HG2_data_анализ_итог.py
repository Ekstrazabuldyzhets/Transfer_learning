import os
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

MAIN_DIRECTORY = "/Users/nierra/Desktop/диплом-2/датасет_2/Data"
LG_HG2_PATH = os.path.join(MAIN_DIRECTORY, "LG_HG2_data")

# Ограничения для вывода
MAX_FILES_DETAIL = 198  # Будем выводить все файлы детально


# ============================================
# ФУНКЦИЯ ПОИСКА СТРОКИ С ЗАГОЛОВКАМИ
# ============================================

def find_header_line(filepath, encoding='utf-8'):
    """
    Находит строку с заголовками столбцов в CSV файле
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'Time Stamp' in line and 'Voltage' in line and 'Current' in line:
                return i
            if 'Step' in line and 'Status' in line and 'Capacity' in line:
                return i

        return None
    except Exception as e:
        return None


# ============================================
# ФУНКЦИЯ ПРАВИЛЬНОЙ ЗАГРУЗКИ CSV
# ============================================

def load_csv_correctly(filepath):
    """
    Загружает CSV файл LG HG2 с правильным пропуском заголовков
    """
    encodings = ['utf-8', 'latin1', 'cp1252']

    for encoding in encodings:
        header_line = find_header_line(filepath, encoding)

        if header_line is not None:
            try:
                df = pd.read_csv(
                    filepath,
                    sep=',',
                    skiprows=header_line,
                    encoding=encoding,
                    low_memory=False
                )

                # Пропускаем строку с единицами измерения
                if len(df) > 0:
                    first_row = df.iloc[0]
                    if any(isinstance(x, str) and '[' in str(x) for x in first_row):
                        df = df.iloc[1:].reset_index(drop=True)

                # Конвертируем числовые колонки
                numeric_columns = ['Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu',
                                   'Step', 'Cycle', 'Cycle Level', 'Cnt']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if len(df) > 0 and len(df.columns) > 3:
                    return df
            except Exception as e:
                pass

    return None


# ============================================
# ФУНКЦИЯ РАСЧЁТА SoC ИЗ CAPACITY
# ============================================

def calculate_soc_from_capacity(capacity_data):
    """
    Рассчитывает SoC (0..1) из Capacity
    """
    if capacity_data is None or len(capacity_data) == 0:
        return None

    try:
        cap_clean = capacity_data.dropna()
        if len(cap_clean) == 0:
            return None

        cap_min = cap_clean.min()
        cap_max = cap_clean.max()

        if cap_max > cap_min:
            soc = (cap_clean - cap_min) / (cap_max - cap_min)
            soc = np.clip(soc, 0.0, 1.0)
            return soc, float(cap_min), float(cap_max)
    except Exception as e:
        return None

    return None


# ============================================
# ФУНКЦИЯ АНАЛИЗА ОДНОГО ФАЙЛА
# ============================================

def analyze_csv_file(filepath, rel_path):
    """
    Полный анализ одного CSV файла
    """
    result = {
        'path': rel_path,
        'full_path': filepath,
        'size_kb': os.path.getsize(filepath) / 1024,
        'num_rows': 0,
        'num_cols': 0,
        'columns': [],
        'has_nan': False,
        'soc_stats': None,
        'temp_folder': rel_path.split(os.sep)[0] if os.sep in rel_path else 'root',
        'file_type': None,
        'error': None,
        'stats': {}
    }

    # Определяем тип файла по имени
    file_name = os.path.basename(filepath).lower()
    if 'c20' in file_name:
        result['file_type'] = 'C20 (калибровка)'
    elif 'hppc' in file_name:
        result['file_type'] = 'HPPC (импульсный)'
    elif 'udds' in file_name:
        result['file_type'] = 'UDDS (городской цикл)'
    elif 'hwfet' in file_name:
        result['file_type'] = 'HWFET (загородный цикл)'
    elif 'la92' in file_name:
        result['file_type'] = 'LA92 (агрессивный цикл)'
    elif 'us06' in file_name:
        result['file_type'] = 'US06 (высокоскоростной цикл)'
    elif 'mixed' in file_name:
        result['file_type'] = 'Mixed (смешанный цикл)'
    elif 'charge' in file_name:
        result['file_type'] = 'Charge (заряд)'
    elif 'dis_' in file_name:
        result['file_type'] = 'Discharge (разряд)'
    elif 'paus' in file_name:
        result['file_type'] = 'Pause (пауза)'
    elif 'cap_' in file_name:
        result['file_type'] = 'Capacity test'
    else:
        result['file_type'] = 'Other'

    df = load_csv_correctly(filepath)

    if df is None:
        result['error'] = "Не удалось загрузить CSV"
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

        # Статистика для числовых колонок
        if pd.api.types.is_numeric_dtype(df[col]):
            clean_data = df[col].dropna()
            if len(clean_data) > 0:
                col_info['min'] = float(clean_data.min())
                col_info['max'] = float(clean_data.max())
                col_info['mean'] = float(clean_data.mean())
                col_info['std'] = float(clean_data.std())

        result['stats'][col] = col_info

    # Расчёт SoC из Capacity
    if 'Capacity' in df.columns and pd.api.types.is_numeric_dtype(df['Capacity']):
        soc_result = calculate_soc_from_capacity(df['Capacity'])
        if soc_result:
            result['soc_stats'] = {
                'source': 'Capacity',
                'cap_min': soc_result[1],
                'cap_max': soc_result[2],
                'soc_min': float(soc_result[0].min()),
                'soc_max': float(soc_result[0].max()),
                'soc_mean': float(soc_result[0].mean())
            }

    return result


# ============================================
# ФУНКЦИЯ ВЫВОДА ИНФОРМАЦИИ О ФАЙЛЕ (ПОЛНЫЙ)
# ============================================

def print_file_analysis_full(result, idx, total):
    """
    Выводит полный анализ одного файла
    """
    print(f"\n{'=' * 120}")
    print(f"[{idx}/{total}] 📄 {result['path']}")
    print(f"{'=' * 120}")
    print(f"   Размер: {result['size_kb']:.1f} KB")
    print(f"   Тип: {result['file_type']}")
    print(f"   Температурная папка: {result['temp_folder']}")

    if result['error']:
        print(f"   ❌ ОШИБКА: {result['error']}")
        return

    print(f"   ✅ УСПЕШНО ЗАГРУЖЕНО")
    print(f"   Строк: {result['num_rows']:,}")
    print(f"   Столбцов: {result['num_cols']}")
    print(f"   Колонки: {', '.join(result['columns'][:12])}")
    if len(result['columns']) > 12:
        print(f"            ... и еще {len(result['columns']) - 12} колонок")

    print(f"\n   📊 СТАТИСТИКА ПО ВСЕМ КОЛОНКАМ:")

    # Сначала выводим ключевые колонки
    important_cols = ['Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Cycle', 'Step', 'Step Time']

    for col in important_cols:
        if col in result['stats']:
            stats = result['stats'][col]
            print(f"\n   🔹 {col}:")
            print(f"      тип: {stats['dtype']}")
            if stats['null_count'] > 0:
                print(f"      пропуски: {stats['null_count']:,} ({stats['null_pct']:.2f}%)")
            if 'min' in stats:
                if col in ['Voltage', 'Current', 'Temperature']:
                    print(f"      min: {stats['min']:.4f}")
                    print(f"      max: {stats['max']:.4f}")
                    print(f"      mean: {stats['mean']:.4f}")
                    print(f"      std: {stats['std']:.4f}")
                elif col in ['Capacity', 'WhAccu']:
                    print(f"      min: {stats['min']:.4f}")
                    print(f"      max: {stats['max']:.4f}")
                    print(f"      mean: {stats['mean']:.4f}")
                else:
                    print(f"      min: {stats['min']:.2f}")
                    print(f"      max: {stats['max']:.2f}")
                    print(f"      mean: {stats['mean']:.2f}")

    # Выводим остальные колонки кратко
    other_cols = [c for c in result['stats'].keys() if c not in important_cols and c != 'Unnamed: 14']
    if other_cols:
        print(f"\n   🔹 Другие колонки:")
        for col in other_cols[:5]:
            stats = result['stats'][col]
            print(f"      • {col}: {stats['dtype']}, {stats['null_count']} пропусков")

    if result.get('soc_stats'):
        print(f"\n   🎯 SoC (рассчитан из {result['soc_stats']['source']}):")
        print(
            f"      Диапазон Capacity: {result['soc_stats']['cap_min']:.4f} – {result['soc_stats']['cap_max']:.4f} А·ч")
        print(f"      SoC: {result['soc_stats']['soc_min'] * 100:.1f}% – {result['soc_stats']['soc_max'] * 100:.1f}%")
        print(f"      Средний SoC: {result['soc_stats']['soc_mean'] * 100:.1f}%")

    if result['has_nan']:
        print(f"\n   ⚠️ ВНИМАНИЕ: в файле есть пропуски (NaN)")


# ============================================
# ФУНКЦИЯ СБОРА ВСЕХ CSV ФАЙЛОВ
# ============================================

def collect_csv_files(base_path):
    """
    Собирает все CSV файлы
    """
    csv_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                csv_files.append((full_path, rel_path))
    return csv_files


# ============================================
# ФУНКЦИЯ СВОДНОЙ СТАТИСТИКИ ПО ВСЕМ ФАЙЛАМ
# ============================================

def print_summary(results, total_found):
    """
    Печатает сводку по всем файлам
    """
    print("\n" + "=" * 120)
    print("📊 СВОДНАЯ СТАТИСТИКА ПО ВСЕМУ ДАТАСЕТУ LG HG2")
    print("=" * 120)

    analyzed = [r for r in results if not r['error']]
    files_with_error = sum(1 for r in results if r['error'])
    files_with_soc = sum(1 for r in results if r.get('soc_stats'))
    total_rows = sum(r['num_rows'] for r in analyzed)

    print(f"\n📁 Всего CSV файлов: {total_found}")
    print(f"✅ Успешно загружено: {len(analyzed)}")
    print(f"❌ С ошибками: {files_with_error}")
    print(f"📏 Всего строк данных: {total_rows:,}")
    print(f"🎯 С рассчитанным SoC: {files_with_soc}")

    # Статистика по типам файлов
    print(f"\n📂 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ ФАЙЛОВ:")
    type_stats = defaultdict(int)
    for r in analyzed:
        type_stats[r['file_type']] += 1

    for file_type, count in sorted(type_stats.items()):
        print(f"   {file_type}: {count} файлов")

    # Статистика по температурам
    print(f"\n🌡️ РАСПРЕДЕЛЕНИЕ ПО ТЕМПЕРАТУРАМ:")
    temp_stats = defaultdict(lambda: {'files': 0, 'rows': 0})
    for r in analyzed:
        temp_stats[r['temp_folder']]['files'] += 1
        temp_stats[r['temp_folder']]['rows'] += r['num_rows']

    for temp, stats in sorted(temp_stats.items()):
        temp_display = temp.replace('degC', '°C').replace('n', '-')
        print(f"   {temp_display}: {stats['files']} файлов, {stats['rows']:,} строк")

    # Общие диапазоны
    all_voltage = []
    all_current = []
    all_temperature = []
    all_capacity = []

    for r in analyzed:
        if 'Voltage' in r['stats'] and 'min' in r['stats']['Voltage']:
            all_voltage.extend([r['stats']['Voltage']['min'], r['stats']['Voltage']['max']])
        if 'Current' in r['stats'] and 'min' in r['stats']['Current']:
            all_current.extend([r['stats']['Current']['min'], r['stats']['Current']['max']])
        if 'Temperature' in r['stats'] and 'min' in r['stats']['Temperature']:
            all_temperature.extend([r['stats']['Temperature']['min'], r['stats']['Temperature']['max']])
        if 'Capacity' in r['stats'] and 'min' in r['stats']['Capacity']:
            all_capacity.extend([r['stats']['Capacity']['min'], r['stats']['Capacity']['max']])

    print(f"\n📈 ОБЩИЕ ДИАПАЗОНЫ ИЗМЕРЕНИЙ:")
    if all_voltage:
        print(f"   Voltage: {min(all_voltage):.2f} – {max(all_voltage):.2f} В")
    if all_current:
        print(f"   Current: {min(all_current):.2f} – {max(all_current):.2f} А")
    if all_temperature:
        print(f"   Temperature: {min(all_temperature):.1f} – {max(all_temperature):.1f} °C")
    if all_capacity:
        print(f"   Capacity: {min(all_capacity):.3f} – {max(all_capacity):.3f} А·ч")


# ============================================
# ФУНКЦИЯ ДЕТАЛЬНОГО АНАЛИЗА ОДНОГО ПРИМЕРНОГО ФАЙЛА
# ============================================

def analyze_example_file(base_path):
    """
    Детальный анализ одного файла (C20 для примера)
    """
    print("\n" + "=" * 120)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА (C20 калибровка)")
    print("=" * 120)

    # Ищем C20 файл
    example_file = None
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'C20DisCh' in file and file.endswith('.csv'):
                example_file = os.path.join(root, file)
                rel_path = os.path.relpath(example_file, base_path)
                break
        if example_file:
            break

    if not example_file:
        print("❌ Не найден C20 файл")
        return

    print(f"\n📄 {rel_path}")
    print(f"Размер: {os.path.getsize(example_file) / 1024:.1f} KB")

    df = load_csv_correctly(example_file)
    if df is None:
        print("❌ Не удалось загрузить")
        return

    print(f"\n✅ УСПЕШНО ЗАГРУЖЕНО")
    print(f"📊 Форма: {df.shape[0]} строк × {df.shape[1]} столбцов")

    print(f"\n📋 НАЗВАНИЯ СТОЛБЦОВ:")
    for i, col in enumerate(df.columns):
        print(f"   [{i}] {col}")

    print(f"\n🔍 ПЕРВЫЕ 10 СТРОК (числовые колонки):")
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].head(10))

    print(f"\n📈 ПОЛНАЯ СТАТИСТИКА ЧИСЛОВЫХ СТОЛБЦОВ:")
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())

    # Информация о качестве данных
    print(f"\n📊 КАЧЕСТВО ДАННЫХ:")
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"   {col}: {null_count} пропусков ({100 * null_count / len(df):.2f}%)")

    # Capacity и SoC
    if 'Capacity' in df.columns and pd.api.types.is_numeric_dtype(df['Capacity']):
        cap_min = df['Capacity'].min()
        cap_max = df['Capacity'].max()
        print(f"\n🎯 CAPACITY И SoC:")
        print(f"   Capacity min: {cap_min:.4f} А·ч")
        print(f"   Capacity max: {cap_max:.4f} А·ч")
        print(f"   Диапазон: {cap_max - cap_min:.4f} А·ч")
        if cap_max > cap_min:
            print(f"   SoC из Capacity: 0% – 100%")


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main():
    if not os.path.exists(LG_HG2_PATH):
        print(f"❌ Путь не найден: {LG_HG2_PATH}")
        return

    print("=" * 120)
    print("🔬 ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА LG HG2 (ПОКАЗАНИЯ ПО КАЖДОМУ ФАЙЛУ)")
    print("=" * 120)
    print(f"\n📁 Путь: {LG_HG2_PATH}")

    # Собираем все CSV файлы
    csv_files = collect_csv_files(LG_HG2_PATH)
    print(f"\n📂 Найдено CSV файлов: {len(csv_files)}")

    # Анализируем каждый файл
    results = []

    for idx, (full_path, rel_path) in enumerate(csv_files, 1):
        result = analyze_csv_file(full_path, rel_path)
        results.append(result)

        # Выводим детальную информацию по каждому файлу
        print_file_analysis_full(result, idx, len(csv_files))

    # Сводная статистика
    print_summary(results, len(csv_files))

    # Детальный анализ примера
    analyze_example_file(LG_HG2_PATH)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ДАТАСЕТА LG HG2 ЗАВЕРШЁН")
    print(f"   Всего проанализировано файлов: {len(results)}")
    print(f"   Успешно: {len([r for r in results if not r['error']])}")
    print(f"   С ошибками: {len([r for r in results if r['error']])}")
    print("=" * 120)


if __name__ == "__main__":
    main()