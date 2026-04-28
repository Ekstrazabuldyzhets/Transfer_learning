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

MAX_FILES_DETAIL = 10


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
                # Загружаем с пропуском строк до заголовков
                df = pd.read_csv(
                    filepath,
                    sep=',',
                    skiprows=header_line,
                    encoding=encoding,
                    low_memory=False
                )

                # Пропускаем строку с единицами измерения (которая может быть после заголовков)
                # Проверяем первую строку данных: если там значения в квадратных скобках - пропускаем
                if len(df) > 0:
                    first_row = df.iloc[0]
                    # Проверяем, содержит ли первая строка единицы измерения
                    if any(isinstance(x, str) and '[' in str(x) for x in first_row):
                        df = df.iloc[1:].reset_index(drop=True)

                # Конвертируем числовые колонки
                numeric_columns = ['Voltage', 'Current', 'Temperature', 'Capacity', 'WhAccu', 'Step', 'Cycle',
                                   'Cycle Level', 'Cnt']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                if len(df) > 0 and len(df.columns) > 3:
                    return df
            except Exception as e:
                pass

    return None


# ============================================
# ФУНКЦИЯ АНАЛИЗА СТРУКТУРЫ ПАПОК
# ============================================

def analyze_folder_structure(base_path):
    """
    Анализирует структуру папок датасета
    """
    print("\n" + "=" * 120)
    print("📁 АНАЛИЗ СТРУКТУРЫ ПАПОК LG HG2")
    print("=" * 120)

    temp_folders = []
    total_csv = 0

    for root, dirs, files in os.walk(base_path):
        level = root.replace(base_path, '').count(os.sep)
        folder_name = os.path.basename(root)

        csv_files = [f for f in files if f.endswith('.csv')]
        total_csv += len(csv_files)

        if level == 0:
            print(f"\n📂 {folder_name}/")
        else:
            indent = '   ' * (level - 1)
            print(f"{indent}📁 {folder_name}/ ({len(csv_files)} CSV файлов)")
            if folder_name not in temp_folders:
                temp_folders.append(folder_name)

    return temp_folders, total_csv


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
        'error': None,
        'stats': {}
    }

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

        # Поиск SoC или Capacity
        col_lower = col.lower()
        if 'soc' in col_lower:
            if pd.api.types.is_numeric_dtype(df[col]):
                result['soc_stats'] = {
                    'source': col,
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'is_direct': True
                }
        elif 'capacity' in col_lower:
            if pd.api.types.is_numeric_dtype(df[col]):
                cap_data = df[col].dropna()
                if len(cap_data) > 0:
                    cap_min = cap_data.min()
                    cap_max = cap_data.max()
                    if cap_max > cap_min:
                        soc_calc = (cap_data - cap_min) / (cap_max - cap_min)
                        result['soc_stats'] = {
                            'source': col,
                            'cap_min': float(cap_min),
                            'cap_max': float(cap_max),
                            'soc_min': float(soc_calc.min()),
                            'soc_max': float(soc_calc.max()),
                            'soc_mean': float(soc_calc.mean()),
                            'is_direct': False
                        }

    return result


# ============================================
# ФУНКЦИЯ ВЫВОДА ИНФОРМАЦИИ О ФАЙЛЕ
# ============================================

def print_file_analysis(result, idx, total):
    """
    Выводит анализ одного файла
    """
    print(f"\n{'=' * 100}")
    print(f"[{idx}/{total}] 📄 {result['path']}")
    print(f"{'=' * 100}")
    print(f"   Размер: {result['size_kb']:.1f} KB")

    if result['error']:
        print(f"   ❌ ОШИБКА: {result['error']}")
        return

    print(f"   ✅ УСПЕШНО ЗАГРУЖЕНО")
    print(f"   Строк: {result['num_rows']:,}")
    print(f"   Столбцов: {result['num_cols']}")
    print(f"   Колонки: {', '.join(result['columns'][:10])}")
    if len(result['columns']) > 10:
        print(f"            ... и еще {len(result['columns']) - 10} колонок")

    print(f"\n   📊 СТАТИСТИКА ПО КЛЮЧЕВЫМ КОЛОНКАМ:")
    important_cols = ['Voltage', 'Current', 'Temperature', 'Capacity', 'Cycle']
    for col in important_cols:
        if col in result['stats'] and 'min' in result['stats'][col]:
            stats = result['stats'][col]
            print(f"\n   🔹 {col}:")
            print(f"      min: {stats['min']:.4f}")
            print(f"      max: {stats['max']:.4f}")
            print(f"      mean: {stats['mean']:.4f}")
        elif col in result['stats']:
            print(f"\n   🔹 {col}: нет числовых данных (тип: {result['stats'][col]['dtype']})")

    if result.get('soc_stats'):
        print(f"\n   🎯 SoC:")
        if result['soc_stats'].get('is_direct'):
            print(f"      (прямое поле: {result['soc_stats']['source']})")
            print(f"      min: {result['soc_stats']['min'] * 100:.1f}%")
            print(f"      max: {result['soc_stats']['max'] * 100:.1f}%")
        else:
            print(f"      (рассчитан из {result['soc_stats']['source']})")
            print(
                f"      SoC: {result['soc_stats']['soc_min'] * 100:.1f}% – {result['soc_stats']['soc_max'] * 100:.1f}%")


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
# СВОДНАЯ СТАТИСТИКА
# ============================================

def print_summary(results, total_found):
    """
    Печатает сводку
    """
    print("\n" + "=" * 120)
    print("📊 СВОДНАЯ СТАТИСТИКА")
    print("=" * 120)

    analyzed = [r for r in results if not r['error']]
    files_with_error = sum(1 for r in results if r['error'])
    files_with_soc = sum(1 for r in results if r.get('soc_stats'))
    total_rows = sum(r['num_rows'] for r in analyzed)

    print(f"\n📁 Всего CSV файлов: {total_found}")
    print(f"✅ Успешно загружено: {len(analyzed)}")
    print(f"❌ С ошибками: {files_with_error}")
    print(f"📏 Всего строк в успешных: {total_rows:,}")
    print(f"🎯 С SoC: {files_with_soc}")

    # Общая статистика по Voltage, Current, Temperature, Capacity
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

    if all_voltage:
        print(f"\n📈 Общий диапазон Voltage: {min(all_voltage):.2f} – {max(all_voltage):.2f} В")
    if all_current:
        print(f"📈 Общий диапазон Current: {min(all_current):.2f} – {max(all_current):.2f} А")
    if all_temperature:
        print(f"📈 Общий диапазон Temperature: {min(all_temperature):.1f} – {max(all_temperature):.1f} °C")
    if all_capacity:
        print(f"📈 Общий диапазон Capacity: {min(all_capacity):.3f} – {max(all_capacity):.3f} А·ч")

    # Распределение по температурам
    print("\n📂 Распределение по температурам:")
    temp_stats = defaultdict(lambda: {'total': 0, 'success': 0})
    for r in results:
        parts = r['path'].split(os.sep)
        if len(parts) >= 1:
            temp = parts[0]
            temp_stats[temp]['total'] += 1
            if not r['error']:
                temp_stats[temp]['success'] += 1

    for temp, stats in sorted(temp_stats.items()):
        print(f"   {temp}: {stats['success']}/{stats['total']} успешно")


# ============================================
# ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРА
# ============================================

def analyze_example_file(base_path):
    """
    Детальный анализ файла C20
    """
    print("\n" + "=" * 120)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА (C20)")
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

    print(f"\n📋 Названия столбцов:")
    for i, col in enumerate(df.columns):
        print(f"   [{i}] {col}")

    print(f"\n🔍 Первые 5 строк данных (числовые колонки):")
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 200)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].head())
    else:
        print(df.head())

    print(f"\n📈 Статистика числовых столбцов:")
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())

    # Проверяем Capacity
    if 'Capacity' in df.columns and pd.api.types.is_numeric_dtype(df['Capacity']):
        cap_min = df['Capacity'].min()
        cap_max = df['Capacity'].max()
        print(f"\n🎯 Capacity (А·ч):")
        print(f"   min: {cap_min:.4f}")
        print(f"   max: {cap_max:.4f}")
        print(f"   range: {cap_max - cap_min:.4f}")
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
    print("🔬 ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА LG HG2")
    print("=" * 120)
    print(f"\n📁 Путь: {LG_HG2_PATH}")

    # 1. Анализ структуры папок
    temp_folders, total_csv = analyze_folder_structure(LG_HG2_PATH)
    print(f"\n📂 Температурные папки: {', '.join(temp_folders)}")
    print(f"📄 Всего CSV файлов: {total_csv}")

    # 2. Собираем файлы
    csv_files = collect_csv_files(LG_HG2_PATH)

    # 3. Анализируем каждый файл
    results = []
    successful = 0

    for idx, (full_path, rel_path) in enumerate(csv_files, 1):
        result = analyze_csv_file(full_path, rel_path)
        results.append(result)

        if not result['error']:
            successful += 1
            if successful <= MAX_FILES_DETAIL:
                print_file_analysis(result, idx, len(csv_files))
        else:
            if idx % 20 == 0:
                print(f"[{idx}/{len(csv_files)}] ⏳ Обработано {idx} файлов, успешно: {successful}")

    # 4. Сводная статистика
    print_summary(results, total_csv)

    # 5. Детальный анализ примера
    analyze_example_file(LG_HG2_PATH)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ДАТАСЕТА LG HG2 ЗАВЕРШЁН")
    print("=" * 120)


if __name__ == "__main__":
    main()