"""
ФИНАЛЬНАЯ ПРЕДОБРАБОТКА ДЛЯ ТРАНСФЕРНОГО ОБУЧЕНИЯ

Что делает:
1. Panasonic → Panasonic_Preprocessed (Current_A, Voltage_V, Temperature_C, SOC_percent)
2. LG HG2 → LG_HG2_Preprocessed (с апсемплингом 1 Гц → 10 Гц)
3. BMW i3 → Audi_Preprocessed (с масками достоверности)

Выходной формат (единый для всех existing кодов):
- Current_A, Voltage_V, Temperature_C, SOC_percent, Mask

Пути для existing кодов:
- PANASONIC_PRE = ".../Panasonic_Preprocessed"
- AUDI_PRE = ".../Audi_Preprocessed"  # это будет BMW i3
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Пути к первично обработанным данным
PATHS = {
    "panasonic": "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/Panasonic_processed",
    "lg_hg2": "/Users/nierra/Desktop/диплом-2/датасет_2/Data/LG_HG2_processed",
    "bmw": "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/BMW_processed"
}

# Пути для сохранения (ожидаемые existing кодами)
OUTPUT_PATHS = {
    "panasonic": "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/Panasonic_Preprocessed_final",
    "lg_hg2": "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/LG_HG2_Preprocessed_final",
    "bmw": "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2/BMW_Preprocessed_final"
}

# Параметры
TARGET_FS = 10.0  # целевая частота дискретизации (Гц)


# ============================================================================
# 1. PANASONIC (лаборатория, NCA)
# ============================================================================

def process_panasonic():
    """Panasonic → Panasonic_Preprocessed_final"""
    print("\n" + "=" * 60)
    print("🔵 PANASONIC → Panasonic_Preprocessed_final")
    print("=" * 60)

    source_dir = PATHS["panasonic"]
    output_dir = OUTPUT_PATHS["panasonic"]
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.endswith('_processed.csv'):
                continue

            filepath = os.path.join(root, file)
            df = pd.read_csv(filepath)

            # Проверка наличия колонок
            if 'Voltage' not in df.columns or 'Current' not in df.columns:
                continue

            # Колонка температуры
            if 'Battery_Temp' in df.columns:
                temp_col = 'Battery_Temp'
            elif 'Temperature' in df.columns:
                temp_col = 'Temperature'
            else:
                continue

            # Формируем выходной DataFrame
            df_out = pd.DataFrame()
            df_out['Current_A'] = df['Current'].values
            df_out['Voltage_V'] = df['Voltage'].values
            df_out['Temperature_C'] = df[temp_col].values

            # SoC (в процентах)
            if 'SoC_fraction' in df.columns:
                df_out['SOC_percent'] = df['SoC_fraction'].values * 100
            elif 'SoC_percent' in df.columns:
                df_out['SOC_percent'] = df['SoC_percent'].values
            elif 'SoC' in df.columns:
                if df['SoC'].max() <= 1:
                    df_out['SOC_percent'] = df['SoC'].values * 100
                else:
                    df_out['SOC_percent'] = df['SoC'].values
            elif 'SOC [-]' in df.columns:
                df_out['SOC_percent'] = df['SOC [-]'].values * 100
            else:
                continue

            df_out['SOC_percent'] = df_out['SOC_percent'].clip(0, 100)
            df_out['Mask'] = 1

            # Сохраняем с сохранением структуры
            rel_path = os.path.relpath(root, source_dir)
            target_dir = os.path.join(output_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)

            output_file = os.path.join(target_dir, file.replace('_processed.csv', '.csv'))
            df_out.to_csv(output_file, index=False)
            count += 1

    print(f"   ✅ Обработано файлов: {count}")
    print(f"   📁 {output_dir}")


# ============================================================================
# 2. LG HG2 (лаборатория, NMC) С АПСЕМПЛИНГОМ
# ============================================================================

def upsample_to_10hz(df, time_col='Time [s]', target_fs=10.0):
    """Апсемплинг с 1 Гц до 10 Гц кубической сплайн-интерполяцией"""
    if time_col not in df.columns:
        return df

    dt_current = df[time_col].diff().median()
    current_fs = 1 / dt_current if dt_current > 0 else 1.0

    # Если уже 10 Гц или близко, возвращаем как есть
    if abs(current_fs - target_fs) < 0.5:
        return df

    t_start = df[time_col].iloc[0]
    t_end = df[time_col].iloc[-1]
    t_new = np.arange(t_start, t_end, 1 / target_fs)

    df_up = pd.DataFrame({time_col: t_new})

    for col in df.columns:
        if col == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and len(df[col]) > 1:
            try:
                interp = interp1d(df[time_col], df[col], kind='cubic',
                                  bounds_error=False, fill_value='extrapolate')
                df_up[col] = interp(t_new)
            except:
                # Если интерполяция не удалась, используем ближайшее значение
                interp = interp1d(df[time_col], df[col], kind='nearest',
                                  bounds_error=False, fill_value='extrapolate')
                df_up[col] = interp(t_new)
        else:
            df_up[col] = df[col].iloc[0] if len(df) > 0 else 0

    return df_up


def process_lg_hg2():
    """LG HG2 → LG_HG2_Preprocessed_final (с апсемплингом 1 Гц → 10 Гц)"""
    print("\n" + "=" * 60)
    print("🟢 LG HG2 → LG_HG2_Preprocessed_final (с апсемплингом)")
    print("=" * 60)

    source_dir = PATHS["lg_hg2"]
    output_dir = OUTPUT_PATHS["lg_hg2"]
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    skipped = 0

    # Обходим все температурные папки (40degC, 25degC, 0degC, n10degC, n20degC)
    for temp_folder in os.listdir(source_dir):
        temp_path = os.path.join(source_dir, temp_folder)
        if not os.path.isdir(temp_path):
            continue

        print(f"\n   📂 Обработка {temp_folder}...")

        # Находим все CSV файлы в этой папке
        csv_files = [f for f in os.listdir(temp_path) if f.endswith('_processed.csv')]

        for file in csv_files:
            filepath = os.path.join(temp_path, file)
            df = pd.read_csv(filepath)

            if len(df) == 0:
                skipped += 1
                continue

            # Проверяем наличие колонки времени
            if 'Time [s]' not in df.columns:
                print(f"   ⚠️ Нет 'Time [s]' в {file}, пропуск")
                skipped += 1
                continue

            # === АПСЕМПЛИНГ до 10 Гц ===
            dt_current = df['Time [s]'].diff().median()
            current_fs = 1 / dt_current if dt_current > 0 else 1.0

            if current_fs < 5.0:  # Нужен апсемплинг
                df = upsample_to_10hz(df, time_col='Time [s]', target_fs=TARGET_FS)
                print(f"      📈 Апсемплинг: {current_fs:.1f} Гц → {TARGET_FS} Гц ({file[:30]}...)")
            else:
                print(f"      ✅ Частота: {current_fs:.1f} Гц ({file[:30]}...)")

            # === ФОРМИРУЕМ ВЫХОДНОЙ DataFrame ===
            df_out = pd.DataFrame()
            df_out['Current_A'] = df['Current [A]'].values if 'Current [A]' in df.columns else df['Current'].values
            df_out['Voltage_V'] = df['Voltage [V]'].values if 'Voltage [V]' in df.columns else df['Voltage'].values
            df_out['Temperature_C'] = df['Temperature [degC]'].values if 'Temperature [degC]' in df.columns else df[
                'Temperature'].values

            # SoC (в процентах)
            if 'SOC [-]' in df.columns:
                df_out['SOC_percent'] = df['SOC [-]'].values * 100
            elif 'SoC_fraction' in df.columns:
                df_out['SOC_percent'] = df['SoC_fraction'].values * 100
            elif 'SoC' in df.columns:
                if df['SoC'].max() <= 1:
                    df_out['SOC_percent'] = df['SoC'].values * 100
                else:
                    df_out['SOC_percent'] = df['SoC'].values
            else:
                print(f"      ⚠️ Нет SoC в {file}, пропуск")
                skipped += 1
                continue

            df_out['SOC_percent'] = df_out['SOC_percent'].clip(0, 100)
            df_out['Mask'] = 1

            # Сохраняем
            target_dir = os.path.join(output_dir, temp_folder)
            os.makedirs(target_dir, exist_ok=True)

            output_file = os.path.join(target_dir, file.replace('_processed.csv', '.csv'))
            df_out.to_csv(output_file, index=False)
            count += 1

    print(f"\n   ✅ Обработано файлов: {count}")
    if skipped > 0:
        print(f"   ⚠️ Пропущено: {skipped}")
    print(f"   📁 {output_dir}")


# ============================================================================
# 3. BMW i3 (полевые данные) С МАСКАМИ
# ============================================================================

def create_bmw_mask(df, velocity_col='Velocity', current_col='Current_A'):
    """Создаёт маску достоверности для BMW i3"""
    if current_col in df.columns:
        current = df[current_col].abs()
    else:
        current = pd.Series([0] * len(df))

    if velocity_col in df.columns:
        velocity = df[velocity_col]
    else:
        velocity = pd.Series([0] * len(df))

    # Условия достоверности
    is_calibration = (current < 1.0) & (velocity == 0)

    mask = is_calibration.astype(int)
    if len(mask) > 0:
        mask.iloc[0] = 1

    return mask


def process_bmw():
    """BMW i3 → BMW_Preprocessed_final (с масками)"""
    print("\n" + "=" * 60)
    print("🚗 BMW i3 → BMW_Preprocessed_final (с масками)")
    print("=" * 60)

    source_dir = PATHS["bmw"]
    output_dir = OUTPUT_PATHS["bmw"]
    os.makedirs(output_dir, exist_ok=True)

    count = 0

    if not os.path.exists(source_dir):
        print(f"   ❌ Папка не найдена: {source_dir}")
        return

    for file in os.listdir(source_dir):
        if not file.endswith('_processed.csv'):
            continue

        filepath = os.path.join(source_dir, file)
        df = pd.read_csv(filepath)

        # Определяем колонки
        if 'Voltage' in df.columns:
            voltage_col = 'Voltage'
        elif 'Battery_Voltage' in df.columns:
            voltage_col = 'Battery_Voltage'
        else:
            continue

        if 'Current' in df.columns:
            current_col = 'Current'
        elif 'Battery_Current' in df.columns:
            current_col = 'Battery_Current'
        else:
            continue

        if 'Temperature' in df.columns:
            temp_col = 'Temperature'
        elif 'Battery_Temperature' in df.columns:
            temp_col = 'Battery_Temperature'
        else:
            continue

        # Формируем выходной DataFrame
        df_out = pd.DataFrame()
        df_out['Current_A'] = df[current_col].values
        df_out['Voltage_V'] = df[voltage_col].values
        df_out['Temperature_C'] = df[temp_col].values

        # SoC
        if 'SoC_BMS_percent' in df.columns:
            df_out['SOC_percent'] = df['SoC_BMS_percent'].values
        elif 'SoC_BMS' in df.columns:
            if df['SoC_BMS'].max() <= 1:
                df_out['SOC_percent'] = df['SoC_BMS'].values * 100
            else:
                df_out['SOC_percent'] = df['SoC_BMS'].values
        elif 'SoC_percent' in df.columns:
            df_out['SOC_percent'] = df['SoC_percent'].values
        else:
            continue

        df_out['SOC_percent'] = df_out['SOC_percent'].clip(0, 100)

        # Скорость для маски
        if 'Velocity' in df.columns:
            df_out['Velocity'] = df['Velocity'].values
        else:
            df_out['Velocity'] = 0

        # Создаём маску
        df_out['Mask'] = create_bmw_mask(df_out, velocity_col='Velocity', current_col='Current_A')
        df_out = df_out.drop(columns=['Velocity'])

        # Сохраняем
        output_file = os.path.join(output_dir, file.replace('_processed.csv', '.csv'))
        df_out.to_csv(output_file, index=False)
        count += 1

    print(f"   ✅ Обработано файлов: {count}")
    print(f"   📁 {output_dir}")


# ============================================================================
# 4. ЗАПУСК ВСЕГО
# ============================================================================

def main():
    print("=" * 70)
    print("ФИНАЛЬНАЯ ПРЕДОБРАБОТКА ДЛЯ ТРАНСФЕРНОГО ОБУЧЕНИЯ")
    print("=" * 70)

    process_panasonic()
    process_lg_hg2()
    process_bmw()

    print("\n" + "=" * 70)
    print("✅ ВСЕ ДАННЫЕ ГОТОВЫ!")
    print("=" * 70)
    print("\n📁 Структура для existing кодов:")
    print(f"   PANASONIC_PRE = \"{OUTPUT_PATHS['panasonic']}\"")
    print(f"   AUDI_PRE = \"{OUTPUT_PATHS['bmw']}\"  # это BMW i3")
    print(f"\n   Формат: Current_A, Voltage_V, Temperature_C, SOC_percent, Mask")
    print(f"   Частота: {TARGET_FS} Гц (везде)")
    print(f"   Маска: 1 — достоверный SoC (калибровка BMS)")


if __name__ == "__main__":
    main()