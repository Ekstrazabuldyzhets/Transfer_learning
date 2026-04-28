# Импорты
import os
import scipy.io as sio
import pandas as pd
from tqdm import tqdm


# =============================================================================
# КОНВЕРТАЦИЯ .mat ФАЙЛОВ PANASONIC В CSV
# Задача: превратить сложные .mat файлы в простые CSV, которые
#         можно обрабатывать твоим существующим кодом для LG
# =============================================================================

def convert_mat_to_csv(mat_file_path: str, csv_file_path: str):
    """Конвертирует один .mat файл Panasonic в CSV"""
    try:
        # Загружаем .mat файл
        mat = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
        meas = mat['meas']

        # Создаем DataFrame (имена колонок как в LG, чтобы код был совместим)
        df = pd.DataFrame({
            'Time Stamp': pd.to_datetime('now'),  # заглушка, т.к. в Panasonic нет Time Stamp
            'Step': 0,  # заглушка
            'Status': 'DCH',  # определим позже по току
            'Prog Time': getattr(meas, 'Time'),  # время в секундах
            'Step Time': getattr(meas, 'Time'),  # время шага
            'Cycle': 0,  # заглушка
            'Cycle Level': 0,  # заглушка
            'Procedure': 'Dynamic',  # заглушка
            'Voltage': getattr(meas, 'Voltage'),  # напряжение [V]
            'Current': getattr(meas, 'Current'),  # ток [A] (+ заряд, - разряд)
            'Temperature': getattr(meas, 'Battery_Temp_degC'),  # температура ячейки
            'Capacity': getattr(meas, 'Ah'),  # КЛЮЧЕВОЕ: кулонометрический интеграл!
            'WhAccu': getattr(meas, 'Wh'),  # энергия [Wh]
            'Cnt': range(len(getattr(meas, 'Time')))  # счётчик
        })

        # Определяем статус (CHG/DCH) по знаку тока
        # Это нужно для совместимости с кодом LG
        df['Status'] = df['Current'].apply(lambda x: 'CHG' if x > 0 else ('DCH' if x < 0 else 'REST'))

        # Форматирование Prog Time как HH:MM:SS (нужно для parse_crude_data из LG)
        total_seconds = df['Prog Time'].astype(int)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        df['Prog Time'] = df['Prog Time'].apply(
            lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}.000"
        )
        df['Step Time'] = df['Prog Time']  # для простоты

        # Сохраняем CSV
        df.to_csv(csv_file_path, index=False)
        return True

    except Exception as e:
        print(f"Ошибка конвертации {mat_file_path}: {e}")
        return False


def convert_all_panasonic_data(input_dir: str, output_dir: str):
    """Конвертирует все .mat файлы из input_dir в CSV в output_dir"""

    # Создаём output директорию если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Собираем все .mat файлы (рекурсивно)
    mat_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mat'):
                rel_path = os.path.relpath(root, input_dir)
                mat_files.append({
                    'mat_path': os.path.join(root, file),
                    'rel_dir': rel_path if rel_path != '.' else '',
                    'mat_name': file.replace('.mat', '')
                })

    print(f"Найдено {len(mat_files)} .mat файлов")

    # Конвертируем каждый файл
    for item in tqdm(mat_files, desc="Конвертация .mat → .csv"):
        # Создаём подпапку в output (сохраняем структуру)
        target_dir = os.path.join(output_dir, item['rel_dir'])
        os.makedirs(target_dir, exist_ok=True)

        # Путь к выходному CSV
        csv_path = os.path.join(target_dir, f"{item['mat_name']}.csv")

        # Конвертируем
        convert_mat_to_csv(item['mat_path'], csv_path)

    print(f"✅ Конвертация завершена! CSV сохранены в {output_dir}")


# =============================================================================
# ЗАПУСК
# =============================================================================
if __name__ == '__main__':
    # Входная директория с .mat файлами Panasonic
    input_directory = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/Panasonic 18650PF Data"

    # Выходная директория для CSV (туда, где твой LG код ожидает данные)
    output_directory = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/Panasonic_as_CSV"

    convert_all_panasonic_data(input_directory, output_directory)