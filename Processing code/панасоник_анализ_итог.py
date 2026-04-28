import os
import numpy as np
import scipy.io as sio
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ
# ============================================

SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 1"
PANASONIC_PATH = os.path.join(SOURCE_PATH, "Panasonic 18650PF Data")


# ============================================
# ФУНКЦИЯ ЗАГРУЗКИ .mat ФАЙЛА (UNIVERSAL)
# ============================================

def load_mat_file(filepath):
    """
    Загружает .mat файл любой версии
    """
    try:
        mat_data = sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        return mat_data
    except Exception as e:
        return None


# ============================================
# ФУНКЦИЯ ИЗВЛЕЧЕНИЯ ДАННЫХ ИЗ STRUCT
# ============================================

def extract_struct_data(struct_obj):
    """
    Рекурсивно извлекает данные из MATLAB struct
    """
    result = {}

    if hasattr(struct_obj, '_fieldnames'):
        for field in struct_obj._fieldnames:
            try:
                value = getattr(struct_obj, field)
                if hasattr(value, '_fieldnames'):
                    result[field] = extract_struct_data(value)
                elif hasattr(value, 'shape'):
                    result[field] = value
                else:
                    result[field] = np.array([value]) if not isinstance(value, (np.ndarray, list)) else value
            except Exception as e:
                result[field] = f"Error: {e}"
    elif isinstance(struct_obj, np.ndarray) and struct_obj.dtype.names:
        for field in struct_obj.dtype.names:
            try:
                value = struct_obj[field]
                result[field] = value
            except Exception as e:
                result[field] = f"Error: {e}"
    else:
        result['data'] = struct_obj

    return result


# ============================================
# ФУНКЦИЯ РАСЧЁТА SoC ИЗ Ah
# ============================================

def calculate_soc_from_ah(ah_data):
    """
    Рассчитывает SoC (0..1) из кулонометрического интеграла Ah
    """
    if ah_data is None or len(ah_data) == 0:
        return None

    try:
        ah_flat = ah_data.flatten()
        ah_max = np.max(ah_flat)
        ah_min = np.min(ah_flat)

        if ah_max > ah_min:
            soc = (ah_flat - ah_min) / (ah_max - ah_min)
        else:
            soc = np.ones_like(ah_flat)

        soc = np.clip(soc, 0.0, 1.0)
        return soc, float(ah_max), float(ah_min)
    except Exception as e:
        return None


# ============================================
# ФУНКЦИЯ ПОЛУЧЕНИЯ СТАТИСТИКИ
# ============================================

def get_stats(data):
    """
    Возвращает статистику для числового массива
    """
    if data is None:
        return None

    try:
        if hasattr(data, 'dtype') and data.dtype.kind in 'fc' and data.size > 0:
            clean = data[np.isfinite(data)]
            if len(clean) > 0:
                return {
                    'min': float(np.min(clean)),
                    'max': float(np.max(clean)),
                    'mean': float(np.mean(clean)),
                    'std': float(np.std(clean)),
                    'size': data.size
                }
    except Exception:
        pass
    return None


# ============================================
# ФУНКЦИЯ АНАЛИЗА ОДНОГО ФАЙЛА
# ============================================

def analyze_single_file(filepath, rel_path):
    """
    Полный анализ одного .mat файла
    """
    result = {
        'path': rel_path,
        'full_path': filepath,
        'size_kb': os.path.getsize(filepath) / 1024,
        'num_points': 0,
        'variables': {},
        'fields': {},
        'has_nan': False,
        'soc_stats': None,
        'error': None
    }

    mat_data = load_mat_file(filepath)
    if mat_data is None:
        result['error'] = "Не удалось загрузить файл"
        return result

    # Проходим по всем переменным в файле
    for key in mat_data.keys():
        if key.startswith('__'):
            continue

        var = mat_data[key]
        result['variables'][key] = type(var).__name__

        # Если переменная - MATLAB struct
        if hasattr(var, '_fieldnames') or (isinstance(var, np.ndarray) and var.dtype.names):
            struct_data = extract_struct_data(var)
            result['fields'][key] = struct_data

            # Анализируем каждое поле структуры
            for field_name, field_data in struct_data.items():
                if isinstance(field_data, np.ndarray):
                    stats = get_stats(field_data)
                    if stats:
                        if field_name.lower() in ['ah', 'charge', 'capacity']:
                            soc_result = calculate_soc_from_ah(field_data)
                            if soc_result:
                                result['soc_stats'] = {
                                    'ah_max': soc_result[1],
                                    'ah_min': soc_result[2],
                                    'soc_min': float(np.min(soc_result[0])),
                                    'soc_max': float(np.max(soc_result[0])),
                                    'soc_mean': float(np.mean(soc_result[0]))
                                }

                        result['num_points'] = max(result['num_points'], stats['size'])

                        # Сохраняем статистику поля
                        if 'stats' not in result:
                            result['stats'] = {}
                        result['stats'][f"{key}.{field_name}"] = stats

        # Если переменная - обычный массив
        elif isinstance(var, np.ndarray) and var.size > 0:
            stats = get_stats(var)
            if stats:
                result['num_points'] = max(result['num_points'], stats['size'])
                if 'stats' not in result:
                    result['stats'] = {}
                result['stats'][key] = stats

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

    print(f"   Точек в файле: {result['num_points']:,}")
    print(f"   Переменные в файле: {list(result['variables'].keys())}")

    if result.get('stats'):
        print(f"\n   📊 СТАТИСТИКА ПОЛЕЙ:")
        for field_name, stats in result['stats'].items():
            print(f"\n   🔹 {field_name}:")
            print(f"      размер: {stats['size']:,}")
            print(f"      min: {stats['min']:.6f}")
            print(f"      max: {stats['max']:.6f}")
            print(f"      mean: {stats['mean']:.6f}")
            print(f"      std: {stats['std']:.6f}")

    if result.get('soc_stats'):
        print(f"\n   🎯 SoC (рассчитан из Ah):")
        print(f"      Ah_max: {result['soc_stats']['ah_max']:.4f} А·ч")
        print(f"      Ah_min: {result['soc_stats']['ah_min']:.4f} А·ч")
        print(f"      SoC: {result['soc_stats']['soc_min'] * 100:.1f}% – {result['soc_stats']['soc_max'] * 100:.1f}%")
        print(f"      SoC_mean: {result['soc_stats']['soc_mean'] * 100:.1f}%")


# ============================================
# ФУНКЦИЯ СБОРА ВСЕХ .mat ФАЙЛОВ
# ============================================

def collect_mat_files(base_path):
    """
    Рекурсивно собирает все .mat файлы
    """
    mat_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.mat'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_path)
                mat_files.append((full_path, rel_path))
    return mat_files


# ============================================
# ФУНКЦИЯ СВОДНОЙ СТАТИСТИКИ
# ============================================

def print_summary(results):
    """
    Печатает сводку по всем файлам
    """
    print("\n" + "=" * 120)
    print("📊 СВОДНАЯ СТАТИСТИКА")
    print("=" * 120)

    total_files = len(results)
    files_with_error = sum(1 for r in results if r['error'])
    files_with_soc = sum(1 for r in results if r.get('soc_stats'))
    total_points = sum(r['num_points'] for r in results)

    print(f"\n📁 Всего файлов: {total_files}")
    print(f"📏 Всего точек: {total_points:,}")
    print(f"❌ С ошибками: {files_with_error}")
    print(f"🎯 С рассчитанным SoC: {files_with_soc}")

    # Группировка по папкам
    print("\n📂 Группировка по папкам:")
    folder_counts = defaultdict(int)
    for r in results:
        parts = r['path'].split(os.sep)
        if len(parts) >= 2:
            folder = f"{parts[0]}/{parts[1]}"
            folder_counts[folder] += 1

    for folder, count in sorted(folder_counts.items()):
        print(f"   {folder}: {count} файлов")


# ============================================
# ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА
# ============================================

def analyze_example_file(base_path):
    """
    Детальный анализ C/20 теста
    """
    print("\n" + "=" * 120)
    print("🔬 ДЕТАЛЬНЫЙ АНАЛИЗ ПРИМЕРНОГО ФАЙЛА (C/20 тест)")
    print("=" * 120)

    example_file = None
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'C20' in file and file.endswith('.mat'):
                example_file = os.path.join(root, file)
                rel_path = os.path.relpath(example_file, base_path)
                break
        if example_file:
            break

    if not example_file:
        print("❌ C/20 файл не найден")
        return

    print(f"\n📄 {rel_path}")
    print(f"Размер: {os.path.getsize(example_file) / 1024:.1f} KB\n")

    mat_data = load_mat_file(example_file)
    if mat_data is None:
        print("❌ Не удалось загрузить")
        return

    for key in mat_data.keys():
        if key.startswith('__'):
            continue

        print(f"\n{'=' * 60}")
        print(f"🔑 Переменная: {key} ({type(mat_data[key]).__name__})")
        print(f"{'=' * 60}")

        var = mat_data[key]

        if hasattr(var, '_fieldnames'):
            print(f"   Поля структуры: {list(var._fieldnames)}")
            for field in var._fieldnames:
                try:
                    val = getattr(var, field)
                    if isinstance(val, np.ndarray):
                        print(f"\n   📊 Поле '{field}':")
                        print(f"      форма: {val.shape}")
                        print(f"      dtype: {val.dtype}")
                        if val.dtype.kind in 'fc' and val.size > 0:
                            clean = val[np.isfinite(val)]
                            if len(clean) > 0:
                                print(f"      min: {np.min(clean):.6f}")
                                print(f"      max: {np.max(clean):.6f}")
                                print(f"      mean: {np.mean(clean):.6f}")
                                print(f"      первые 5 значений: {clean[:5]}")
                except Exception as e:
                    print(f"   Ошибка поля {field}: {e}")


# ============================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================

def main():
    """
    Главная функция
    """
    if not os.path.exists(PANASONIC_PATH):
        print(f"❌ Путь не найден: {PANASONIC_PATH}")
        return

    print("=" * 120)
    print("🔬 ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА PANASONIC")
    print("=" * 120)
    print(f"\n📁 Путь: {PANASONIC_PATH}")

    mat_files = collect_mat_files(PANASONIC_PATH)
    print(f"\n📂 Найдено .mat файлов: {len(mat_files)}")

    results = []
    for idx, (full_path, rel_path) in enumerate(mat_files, 1):
        result = analyze_single_file(full_path, rel_path)
        results.append(result)
        print_file_analysis(result, idx, len(mat_files))

    print_summary(results)
    analyze_example_file(PANASONIC_PATH)

    print("\n" + "=" * 120)
    print("✅ АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 120)


# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    main()