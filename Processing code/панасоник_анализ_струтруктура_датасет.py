import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
from collections import defaultdict

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 1"
PANASONIC_PATH = os.path.join(SOURCE_PATH, "Panasonic 18650PF Data")

# Ограничение на количество строк для вывода статистики (чтобы не перегружать)
PREVIEW_MAX_ROWS = 10


# ============================================
# ФУНКЦИЯ ЗАГРУЗКИ .mat ФАЙЛА (ЛЮБАЯ ВЕРСИЯ)
# ============================================

def load_mat_safe(filepath):
    """
    Загружает .mat файл, работая как со старой версией (< v7.3), так и с новой (HDF5).
    Возвращает словарь с данными.
    """
    try:
        # Пробуем стандартную загрузку (для файлов без HDF5)
        data = loadmat(filepath, struct_as_record=False, squeeze_me=True)
        return data
    except NotImplementedError:
        # Файл в формате HDF5 (v7.3)
        try:
            h5_data = {}
            with h5py.File(filepath, 'r') as f:
                for key in f.keys():
                    if not key.startswith('#'):
                        # Читаем и конвертируем в numpy массив
                        val = f[key][()]
                        if isinstance(val, h5py.Dataset):
                            # Если массив байтов, декодируем
                            if val.dtype.kind == 'O' or val.dtype.kind == 'S':
                                # Пытаемся преобразовать в строку
                                try:
                                    val = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in val])
                                except:
                                    pass
                            h5_data[key] = val
            return h5_data
        except Exception as e:
            print(f"   ❌ Ошибка загрузки HDF5: {e}")
            return None


# ============================================
# ФУНКЦИЯ АНАЛИЗА ОДНОГО ФАЙЛА
# ============================================

def analyze_mat_file(filepath, rel_path):
    """
    Полный анализ одного .mat файла.
    Возвращает словарь с результатами.
    """
    result = {
        'path': rel_path,
        'full_path': filepath,
        'size_kb': os.path.getsize(filepath) / 1024,
        'variables': [],
        'error': None,
        'has_nan': False,
        'has_inf': False
    }

    data = load_mat_safe(filepath)
    if data is None:
        result['error'] = "Не удалось загрузить файл"
        return result

    # Анализируем каждую переменную в файле
    for key in data.keys():
        if key.startswith('__'):  # Пропускаем внутренние поля matlab
            continue

        var = data[key]
        var_info = {
            'name': key,
            'type': str(type(var)),
            'shape': var.shape if hasattr(var, 'shape') else 'unknown',
            'dtype': str(var.dtype) if hasattr(var, 'dtype') else 'unknown',
            'size_bytes': var.nbytes if hasattr(var, 'nbytes') else 0,
            'is_struct': False,
            'fields': [],
            'columns': [],
            'stats': None
        }

        # Если переменная является структурой (dtype=object или struct)
        if hasattr(var, 'dtype') and var.dtype.names is not None:
            var_info['is_struct'] = True
            var_info['fields'] = list(var.dtype.names)

            # Пробуем распаковать поля
            for field in var_info['fields']:
                field_data = var[field]
                if hasattr(field_data, 'shape'):
                    var_info['columns'].append({
                        'name': field,
                        'shape': field_data.shape,
                        'dtype': str(field_data.dtype),
                        'min': np.nanmin(field_data) if field_data.size > 0 else None,
                        'max': np.nanmax(field_data) if field_data.size > 0 else None,
                        'mean': np.nanmean(
                            field_data) if field_data.size > 0 and field_data.dtype.kind in 'fc' else None,
                        'has_nan': np.any(pd.isna(field_data)) if field_data.size > 0 else False
                    })
                    if var_info['columns'][-1].get('has_nan', False):
                        result['has_nan'] = True

        # Если переменная является двумерным массивом (таблица)
        elif hasattr(var, 'shape') and len(var.shape) == 2 and var.size > 0:
            try:
                # Преобразуем в pandas DataFrame для анализа
                if var.dtype.kind in 'fc':
                    df = pd.DataFrame(var)
                elif var.dtype.kind == 'O':
                    # Массив объектов - пробуем конвертировать
                    df = pd.DataFrame(var.reshape(var.shape[0], -1))
                else:
                    df = pd.DataFrame(var)

                var_info['columns'] = []
                for col_idx in range(min(df.shape[1], 20)):  # до 20 колонок
                    col_data = df.iloc[:, col_idx]
                    col_info = {
                        'name': f'col_{col_idx}',
                        'shape': col_data.shape,
                        'dtype': str(col_data.dtype),
                        'min': col_data.min() if col_data.dtype.kind in 'fc' else None,
                        'max': col_data.max() if col_data.dtype.kind in 'fc' else None,
                        'mean': col_data.mean() if col_data.dtype.kind in 'fc' else None,
                        'has_nan': col_data.isna().any()
                    }
                    var_info['columns'].append(col_info)
                    if col_info['has_nan']:
                        result['has_nan'] = True

                # Простая статистика
                var_info['stats'] = {
                    'rows': df.shape[0],
                    'cols': df.shape[1],
                    'total_cells': df.shape[0] * df.shape[1]
                }

            except Exception as e:
                var_info['error'] = str(e)

        # Если переменная является вектором
        elif hasattr(var, 'shape') and len(var.shape) == 1 and var.size > 0:
            var_info['stats'] = {
                'length': var.shape[0],
                'min': np.nanmin(var) if var.dtype.kind in 'fc' else None,
                'max': np.nanmax(var) if var.dtype.kind in 'fc' else None,
                'mean': np.nanmean(var) if var.dtype.kind in 'fc' else None,
                'has_nan': np.any(pd.isna(var))
            }
            if var_info['stats']['has_nan']:
                result['has_nan'] = True

        result['variables'].append(var_info)

    return result


# ============================================
# ФУНКЦИЯ ОБХОДА ВСЕХ .mat ФАЙЛОВ
# ============================================

def analyze_all_files(base_path):
    """
    Рекурсивно обходит все .mat файлы и анализирует каждый.
    """
    print("=" * 120)
    print("ПОЛНЫЙ АНАЛИЗ СТРУКТУРЫ ДАТАСЕТА PANASONIC")
    print("=" * 120)

    results = []
    mat_files = []

    # Собираем все .mat файлы
    print("\n📂 Поиск .mat файлов...")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.mat'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                mat_files.append((full_path, rel_path))

    print(f"✅ Найдено {len(mat_files)} .mat файлов\n")

    # Анализируем каждый файл
    for idx, (full_path, rel_path) in enumerate(mat_files):
        print(f"\n[{idx + 1}/{len(mat_files)}] 📄 {rel_path}")
        print(f"    Размер: {os.path.getsize(full_path) / 1024:.1f} KB")

        result = analyze_mat_file(full_path, rel_path)

        if result['error']:
            print(f"    ❌ Ошибка: {result['error']}")
        else:
            print(f"    📊 Переменных в файле: {len(result['variables'])}")
            for var in result['variables']:
                print(f"        • {var['name']} | тип: {var['type'][:50]} | форма: {var['shape']}")

                # Если есть поля структуры
                if var.get('fields'):
                    print(f"          Поля: {', '.join(var['fields'][:10])}")
                    if len(var['fields']) > 10:
                        print(f"          ... и еще {len(var['fields']) - 10} полей")

                # Если есть колонки/статистика
                if var.get('stats'):
                    if 'rows' in var['stats']:
                        print(f"          Строк: {var['stats']['rows']}, столбцов: {var['stats']['cols']}")
                    elif 'length' in var['stats']:
                        print(f"          Длина: {var['stats']['length']}")
                        if var['stats']['min'] is not None:
                            print(
                                f"          min: {var['stats']['min']:.4f}, max: {var['stats']['max']:.4f}, mean: {var['stats']['mean']:.4f}")

                # Показываем 5 первых колонок
                if var.get('columns') and len(var['columns']) > 0:
                    print(f"          Первые {min(5, len(var['columns']))} колонок:")
                    for col in var['columns'][:5]:
                        nan_flag = " [ЕСТЬ NaN]" if col.get('has_nan') else ""
                        if col['min'] is not None:
                            print(
                                f"            - {col['name']}: {col['dtype']}, min={col['min']:.4f}, max={col['max']:.4f}{nan_flag}")
                        else:
                            print(f"            - {col['name']}: {col['dtype']}{nan_flag}")

            if result['has_nan']:
                print(f"    ⚠️  ВНИМАНИЕ: обнаружены пропуски (NaN)")

        results.append(result)

    return results


# ============================================
# СВОДНАЯ СТАТИСТИКА
# ============================================

def print_summary(results):
    """
    Печатает сводку по всем проанализированным файлам.
    """
    print("\n" + "=" * 120)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 120)

    total_files = len(results)
    files_with_nan = sum(1 for r in results if r.get('has_nan', False))
    files_with_error = sum(1 for r in results if r.get('error'))

    print(f"\n📁 Всего файлов: {total_files}")
    print(f"⚠️  Файлов с NaN/пропусками: {files_with_nan}")
    print(f"❌ Файлов с ошибками загрузки: {files_with_error}")

    # Группировка по папкам
    print("\n📂 Группировка по папкам:")
    folder_counts = defaultdict(int)
    for r in results:
        parts = r['path'].split(os.sep)
        if len(parts) >= 2:
            folder_key = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else parts[0]
            folder_counts[folder_key] += 1

    for folder, count in sorted(folder_counts.items()):
        print(f"   {folder}: {count} файлов")

    # Поиск файлов с эталонным SoC
    print("\n🎯 ПОИСК ФАЙЛОВ С SoC / OCV:")
    for r in results:
        has_soc = False
        has_ocv = False
        for var in r['variables']:
            name_lower = var['name'].lower()
            if 'soc' in name_lower or 'state' in name_lower:
                has_soc = True
            if 'ocv' in name_lower or 'open_circuit' in name_lower:
                has_ocv = True
        if has_soc or has_ocv:
            print(f"   📄 {r['path']}")
            if has_soc:
                print(f"      - содержит SoC")
            if has_ocv:
                print(f"      - содержит OCV")

    # Поиск файлов с температурой
    print("\n🌡️ ФАЙЛЫ С ТЕМПЕРАТУРОЙ:")
    for r in results:
        for var in r['variables']:
            name_lower = var['name'].lower()
            if 'temp' in name_lower or 'temperature' in name_lower or 't_' in name_lower:
                print(f"   📄 {r['path']} -> переменная: {var['name']}")
                break


# ============================================
# АНАЛИЗ ОДНОГО КОНКРЕТНОГО ФАЙЛА (ДЛЯ ПРИМЕРА)
# ============================================

def analyze_specific_example(base_path):
    """
    Детальный анализ примера файла (например, C/20 теста).
    """
    print("\n" + "=" * 120)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА (C/20 тест)")
    print("=" * 120)

    # Ищем C/20 файл
    c20_file = None
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'C20' in file and file.endswith('.mat'):
                c20_file = os.path.join(root, file)
                rel_path = os.path.relpath(c20_file, base_path)
                break
        if c20_file:
            break

    if not c20_file:
        print("❌ C/20 файл не найден")
        return

    print(f"\n📄 Файл: {rel_path}")
    print(f"Размер: {os.path.getsize(c20_file) / 1024:.1f} KB\n")

    data = load_mat_safe(c20_file)
    if data is None:
        print("❌ Не удалось загрузить")
        return

    for key in data.keys():
        if key.startswith('__'):
            continue

        print(f"\n🔑 Переменная: {key}")
        var = data[key]
        print(f"   Тип: {type(var)}")

        if hasattr(var, 'shape'):
            print(f"   Форма: {var.shape}")
        if hasattr(var, 'dtype'):
            print(f"   dtype: {var.dtype}")

        # Если это структура с полями
        if hasattr(var, 'dtype') and var.dtype.names is not None:
            print(f"   Поля: {list(var.dtype.names)}")
            for field in var.dtype.names:
                field_data = var[field]
                print(f"\n   📊 Поле '{field}':")
                print(f"      форма: {field_data.shape}")
                print(f"      dtype: {field_data.dtype}")
                if field_data.size > 0 and field_data.dtype.kind in 'fc':
                    print(f"      min: {np.nanmin(field_data):.6f}")
                    print(f"      max: {np.nanmax(field_data):.6f}")
                    print(f"      mean: {np.nanmean(field_data):.6f}")
                    nan_count = np.isnan(field_data).sum()
                    if nan_count > 0:
                        print(f"      ⚠️ NaN: {nan_count} ({100 * nan_count / field_data.size:.2f}%)")

        # Если это обычная матрица
        elif hasattr(var, 'shape') and len(var.shape) == 2 and var.size > 0:
            print(f"\n   📊 Матрица {var.shape[0]}×{var.shape[1]}")
            # Показываем первые строки
            n_show = min(PREVIEW_MAX_ROWS, var.shape[0])
            print(f"   Первые {n_show} строк (первые 5 колонок):")
            for i in range(n_show):
                row_str = "      "
                for j in range(min(5, var.shape[1])):
                    val = var[i, j]
                    if isinstance(val, (int, float)):
                        row_str += f"{val:8.4f} "
                    else:
                        row_str += f"{str(val)[:8]} "
                print(row_str)


# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    if not os.path.exists(PANASONIC_PATH):
        print(f"❌ Путь не найден: {PANASONIC_PATH}")
        exit(1)

    print(f"\n📁 Датасет: {PANASONIC_PATH}")

    # 1. Полный анализ всех файлов
    results = analyze_all_files(PANASONIC_PATH)

    # 2. Сводная статистика
    print_summary(results)

    # 3. Детальный анализ примера (C/20)
    analyze_specific_example(PANASONIC_PATH)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 120)