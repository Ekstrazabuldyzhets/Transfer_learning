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
# ФУНКЦИЯ ПРАВИЛЬНОЙ ЗАГРУЗКИ CSV
# ============================================

def load_csv_correctly(filepath):
    """
    Загружает CSV файл LG HG2 с правильными параметрами
    """
    # Пробуем разные варианты
    attempts = [
        # Вариант 1: разделитель ;, первая строка - заголовки
        lambda: pd.read_csv(filepath, sep=';', encoding='latin1', low_memory=False),
        # Вариант 2: разделитель ;, нет заголовков
        lambda: pd.read_csv(filepath, sep=';', header=None, encoding='latin1', low_memory=False),
        # Вариант 3: разделитель ;, пропустить первые строки
        lambda: pd.read_csv(filepath, sep=';', skiprows=1, encoding='latin1', low_memory=False),
        # Вариант 4: разделитель , (стандартный)
        lambda: pd.read_csv(filepath, encoding='latin1', low_memory=False),
        # Вариант 5: автоопределение разделителя
        lambda: pd.read_csv(filepath, sep=None, engine='python', encoding='latin1', low_memory=False)
    ]

    for attempt in attempts:
        try:
            df = attempt()
            # Если получили осмысленный DataFrame, возвращаем
            if df is not None and len(df.columns) > 1 and len(df) > 0:
                # Проверяем, что колонки имеют осмысленные имена
                if not df.iloc[0].astype(str).str.contains('Time Stamp', case=False).any():
                    # Возможно, заголовки сдвинуты
                    pass
                return df
        except Exception as e:
            continue

    # Специальная попытка: прочитать как сырой текст и разобрать вручную
    try:
        with open(filepath, 'r', encoding='latin1') as f:
            lines = f.readlines()

        # Находим строку с заголовками (содержит "Time Stamp")
        header_line_idx = None
        for i, line in enumerate(lines):
            if 'Time Stamp' in line or 'Voltage' in line or 'Current' in line:
                header_line_idx = i
                break

        if header_line_idx is not None:
            # Определяем разделитель
            if ';' in lines[header_line_idx]:
                sep = ';'
            else:
                sep = ','

            # Читаем с правильной строки заголовка
            df = pd.read_csv(filepath, sep=sep, skiprows=header_line_idx, encoding='latin1', low_memory=False)
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
            temp_folders.append(folder_name)

            # Показываем примеры файлов
            if csv_files:
                indent2 = '   ' * level
                print(f"{indent2}   📄 Типы файлов:")
                types = set()
                for f in csv_files:
                    f_lower = f.lower()
                    if 'c20' in f_lower:
                        types.add('C20')
                    elif 'hppc' in f_lower:
                        types.add('HPPC')
                    elif 'udds' in f_lower:
                        types.add('UDDS')
                    elif 'hwfet' in f_lower:
                        types.add('HWFET')
                    elif 'la92' in f_lower:
                        types.add('LA92')
                    elif 'us06' in f_lower:
                        types.add('US06')
                    elif 'mixed' in f_lower:
                        types.add('Mixed')
                    elif 'charge' in f_lower:
                        types.add('Charge')
                    elif 'dis_' in f_lower:
                        types.add('Discharge')
                    elif 'paus' in f_lower:
                        types.add('Pause')
                    elif 'cap_' in f_lower:
                        types.add('Capacity test')
                for t in sorted(types):
                    print(f"{indent2}      - {t}")

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
        elif 'capacity' in col_lower or 'ah' in col_lower:
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

    print(f"   Строк: {result['num_rows']:,}")
    print(f"   Столбцов: {result['num_cols']}")
    print(f"   Колонки: {', '.join(result['columns'][:8])}")
    if len(result['columns']) > 8:
        print(f"            ... и еще {len(result['columns']) - 8} колонок")

    print(f"\n   📊 СТАТИСТИКА ПО КОЛОНКАМ (первые 8):")
    for col, stats in list(result['stats'].items())[:8]:
        print(f"\n   🔹 {col}:")
        print(f"      тип: {stats['dtype']}")
        if stats['null_count'] > 0:
            print(f"      пустых: {stats['null_count']} ({100 * stats['null_count'] / result['num_rows']:.2f}%)")
        if 'min' in stats:
            print(f"      min: {stats['min']:.6f}")
            print(f"      max: {stats['max']:.6f}")
            print(f"      mean: {stats['mean']:.6f}")
            print(f"      std: {stats['std']:.6f}")

    if result.get('soc_stats'):
        print(f"\n   🎯 SoC:")
        if result['soc_stats'].get('is_direct'):
            print(f"      (прямое поле: {result['soc_stats']['source']})")
            print(f"      min: {result['soc_stats']['min'] * 100:.1f}%")
            print(f"      max: {result['soc_stats']['max'] * 100:.1f}%")
            print(f"      mean: {result['soc_stats']['mean'] * 100:.1f}%")
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


# ============================================
# ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРА
# ============================================

def analyze_example_file(base_path):
    """
    Детальный анализ одного файла
    """
    print("\n" + "=" * 120)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА")
    print("=" * 120)

    # Ищем небольшой файл для анализа
    example_file = None
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv') and 'C20' in file:
                example_file = os.path.join(root, file)
                rel_path = os.path.relpath(example_file, base_path)
                break
        if example_file:
            break

    if not example_file:
        print("❌ Не найден подходящий файл")
        return

    print(f"\n📄 {rel_path}")
    print(f"Размер: {os.path.getsize(example_file) / 1024:.1f} KB")

    df = load_csv_correctly(example_file)
    if df is None:
        print("❌ Не удалось загрузить")
        return

    print(f"\n📊 Форма: {df.shape[0]} строк × {df.shape[1]} столбцов")
    print(f"\n📋 Названия столбцов:")
    for i, col in enumerate(df.columns):
        print(f"   [{i}] {col}")

    print(f"\n📊 Типы данных:")
    for col in df.columns:
        print(f"   {col}: {df[col].dtype}")

    print(f"\n🔍 Первые 5 строк:")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df.head())

    print(f"\n📈 Базовая статистика для числовых столбцов:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("   Нет числовых столбцов")


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
            # Краткий вывод для ошибочных
            if idx % 20 == 0:
                print(f"[{idx}/{len(csv_files)} ⏳ Обработано {idx} файлов, успешно: {successful}")

    # 4. Сводная статистика
    print_summary(results, total_csv)

    # 5. Детальный анализ примера
    analyze_example_file(LG_HG2_PATH)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 120)


if __name__ == "__main__":
    main()