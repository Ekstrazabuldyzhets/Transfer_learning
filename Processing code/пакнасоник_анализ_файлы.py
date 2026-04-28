import os
import scipy.io as sio

# Укажите путь к ЛЮБОМУ файлу с ездовым циклом, например, к 25°C
test_file_path = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 1/Panasonic 18650PF Data/25degC/Drive cycles/03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat"


def inspect_meas_structure(file_path):
    """Загружает файл и полностью выводит содержимое meas"""
    print(f"📁 Анализируем файл:\n   {os.path.basename(file_path)}\n")

    # 1. Загружаем файл
    try:
        mat_data = sio.loadmat(file_path)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # 2. Получаем структуру 'meas'
    if 'meas' not in mat_data:
        print("❌ Ключ 'meas' не найден в файле.")
        print(f"Доступные ключи: {list(mat_data.keys())}")
        return

    meas = mat_data['meas']
    print("🔬 Содержимое переменной 'meas':\n")

    # 3. Если это структура (как и ожидалось)
    if meas.dtype.names:
        print("📋 Это структура со следующими полями:")
        for field_name in meas.dtype.names:
            field_data = meas[field_name][0, 0]  # Достаем данные первого элемента

            # Получаем информацию о поле
            if hasattr(field_data, 'shape'):
                shape = field_data.shape
                dtype = field_data.dtype

                # Пытаемся посчитать статистику для числовых массивов
                if field_data.size > 0 and 'float' in str(dtype):
                    print(f"   🔹 {field_name}: shape={shape}, dtype={dtype}")
                    print(
                        f"       min={field_data.min():.3f}, max={field_data.max():.3f}, mean={field_data.mean():.3f}")
                else:
                    print(f"   🔹 {field_name}: shape={shape}, dtype={dtype}")
            else:
                print(f"   🔹 {field_name}: {type(field_data)}")
    else:
        # На всякий случай, если это не структура
        print("⚠️ 'meas' не является структурой. Показываем содержимое:")
        print(meas)

    print("\n" + "=" * 60)
    print("✅ Анализ структуры 'meas' завершен.")
    print("Теперь вы знаете, как называются столбцы с током, напряжением и т.д.")


# Запускаем анализ
inspect_meas_structure(test_file_path)