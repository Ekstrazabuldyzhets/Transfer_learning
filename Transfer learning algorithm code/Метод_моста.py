import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2"
# После финальной предобработки
LG_HG2_PRE = os.path.join(SOURCE_PATH, "Panasonic_Preprocessed_final")
BMW_PRE = os.path.join(SOURCE_PATH, "BMW_Preprocessed_final")

WINDOW_SIZE = 100
STRIDE = 20
MAX_SOURCE = 50000  # Ограничиваем source
TARGET_AUGMENT = 5  # Аугментация target

# функция извлекающая 21 статистический признак из окна
def extract_features(window):
    cur = window[:, 0]
    volt = window[:, 1]
    temp = window[:, 2]
    power = cur * volt

    return np.array([
        np.mean(cur), np.std(cur), np.min(cur), np.max(cur),
        np.percentile(cur, 25), np.percentile(cur, 75),
        np.sum(np.abs(np.diff(cur))),
        np.mean(volt), np.std(volt), np.min(volt), np.max(volt),
        np.percentile(volt, 25), np.percentile(volt, 75), volt[-1] - volt[0],
        np.mean(temp), np.std(temp), np.min(temp), np.max(temp), temp[-1] - temp[0],
        np.mean(power), np.std(power)
    ], dtype=np.float32)

# функция LG HG2 (совместимый формат)
def load_lg_hg2_data(max_samples=None):
    if not os.path.exists(LG_HG2_PRE):
        print(f" Папка не найдена: {LG_HG2_PRE}")
        return np.array([]), np.array([])

    # Находим все папки с температурами
    temp_folders = [f for f in os.listdir(LG_HG2_PRE) if os.path.isdir(os.path.join(LG_HG2_PRE, f))]
    print(f"\n Загрузка panasonic (source)...")
    print(f"   Температур: {temp_folders}")

    X, y = [], []
    total_windows = 0

    # Для каждой температурной папки находим все CSV файлы, оканчивающиеся на .csv.
    for temp_folder in tqdm(temp_folders, desc="Обработка"):
        folder_path = os.path.join(LG_HG2_PRE, temp_folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))

        # Исключаем зарядные и паузы (неестественные паттерны)
        files = [f for f in files if 'Charge' not in f and 'Paus' not in f]

        for file_path in files:
            try:
                df = pd.read_csv(file_path)

                # КЛЮЧЕВОЕ: переименовываем колонки под формат Panasonic
                # и сразу нарезаем на окна с извлечением признаков
                # SoC: переводим проценты в доли
                feat = df[['Current_A', 'Voltage_V', 'Temperature_C']].values.astype(np.float32)
                tgt = df['SOC_percent'].values.astype(np.float32) / 100.0

                # Нарезаем на окна
                for i in range(0, len(feat) - WINDOW_SIZE, STRIDE):
                    if max_samples and total_windows >= max_samples:
                        break

                    window = feat[i:i + WINDOW_SIZE]
                    target = tgt[i + WINDOW_SIZE - 1]

                    if np.isnan(window).any() or np.isnan(target):
                        continue

                    # Извлекаем 21 признак
                    X.append(extract_features(window))
                    y.append(target)
                    total_windows += 1

            except Exception as e:
                continue

    X = np.array(X)
    y = np.array(y)

    print(f"\n  Загружено окон: {len(X)}")
    print(f"    Признаков: {X.shape[1]}")
    print(f"    Диапазон SoC: {y.min() * 100:.0f}% – {y.max() * 100:.0f}%")

    return X, y

def load_bmw_data(max_samples=None, augment_factor=1):
    if not os.path.exists(BMW_PRE):
        print(f" Папка не найдена: {BMW_PRE}")
        return np.array([]), np.array([]), np.array([])

    files = sorted(glob.glob(os.path.join(BMW_PRE, "Trip*.csv")))

    X, y, masks = [], [], []
    count = 0

    for f in tqdm(files, desc="Загрузка BMW i3"):
        df = pd.read_csv(f)
        feat = df[['Current_A', 'Voltage_V', 'Temperature_C']].values.astype(np.float32)
        tgt = df['SOC_percent'].values.astype(np.float32) / 100.0
        msk = df['Mask'].values.astype(bool)

        for i in range(0, len(feat) - WINDOW_SIZE, STRIDE):
            if max_samples and count >= max_samples:
                break

            if not msk[i + WINDOW_SIZE - 1]:
                continue

            window = feat[i:i + WINDOW_SIZE]
            if np.isnan(window).any():
                continue

            X.append(extract_features(window))
            y.append(tgt[i + WINDOW_SIZE - 1])
            masks.append(1)
            count += 1

            # Аугментация
            if augment_factor > 1:
                for _ in range(augment_factor - 1):
                    noise = np.random.normal(0, 0.01, window.shape)
                    X.append(extract_features(window + noise))
                    y.append(tgt[i + WINDOW_SIZE - 1])
                    masks.append(1)
                    count += 1

        if max_samples and count >= max_samples:
            break

    return np.array(X), np.array(y), np.array(masks, dtype=bool)

print("Загрузка данных")

X_source, y_source = load_lg_hg2_data(max_samples=MAX_SOURCE)
X_target, y_target, masks_target = load_bmw_data(max_samples=None, augment_factor=TARGET_AUGMENT)

print(f"\n Данные загружены:")
print(f"   Source: {X_source.shape} (21 признак)")
print(f"   Target: {X_target.shape} (21 признак)")
print(f"   Target с маской: {masks_target.sum()}")

# 2. нормализируем каждый домен отдельно
print("\n Нормализация")

# Нормализуем source
scaler_source = StandardScaler()
X_source_norm = scaler_source.fit_transform(X_source)

# Нормализуем target отдельно
scaler_target = StandardScaler()
X_target_norm = scaler_target.fit_transform(X_target)

print(f"Source: mean={X_source_norm.mean():.3f}, std={X_source_norm.std():.3f}")
print(f"Target: mean={X_target_norm.mean():.3f}, std={X_target_norm.std():.3f}")

# 3. создаем пары для моста
print("Создание пар (BMW -> Panasonic)")
def create_pairs(X_source, y_source, X_target, y_target, n_neighbors=5):
    X_field, X_lab = [], []
    # Проходим по каждому окну BMW
    for x_t, y_t in zip(X_target, y_target):
        # Вычисляем разницу SoC между текущим окном BMW и всеми окнами Panasonic
        diff = np.abs(y_source - y_t)
        # Находим индексы n_neighbors окон Panasonic с наименьшей разницей SoC
        # argsort возвращает индексы, [:n_neighbors] берет первые n_neighbors
        closest = np.argsort(diff)[:n_neighbors]

        # Для каждого найденного ближайшего окна Panasonic создаем пару
        for idx in closest:
            X_field.append(x_t)  # Признаки Audi (вход для моста)
            X_lab.append(X_source[idx])  # Признаки Panasonic (цель для моста)

    return np.array(X_field), np.array(X_lab)

X_field, X_lab = create_pairs(X_source_norm, y_source, X_target_norm, y_target, n_neighbors=5)

print(f"Создано пар: {len(X_field)}")
print(f"X_field (BMW): {X_field.shape}")
print(f"X_lab (LG HG2): {X_lab.shape}")

# Разделение на пары
X_field_train, X_field_test, X_lab_train, X_lab_test = train_test_split(
    X_field, X_lab, test_size=0.2, random_state=42)

# 4. обучение модели-моста с нормализацией
print("обучение модели-моста")

# Нормализация для моста
scaler_field = StandardScaler()
scaler_lab = StandardScaler()

X_field_norm = scaler_field.fit_transform(X_field_train)
X_lab_norm = scaler_lab.fit_transform(X_lab_train)

bridge = RandomForestRegressor(n_estimators=100,
                               max_depth=20,
                               min_samples_split=5,
                               random_state=42,
                               n_jobs=-1)
# Задача: по признакам BMW i3 предсказать признаки Panasonic
bridge.fit(X_field_norm, X_lab_norm)

# Оценка качества моста
X_field_test_norm = scaler_field.transform(X_field_test)
# Предсказываем нормализованные признаки Panasonic
X_lab_pred_norm = bridge.predict(X_field_test_norm)
# Возвращаем предсказания в исходный масштаб (обратное преобразование)
# reshape(-1, X_lab.shape[1]) - приводим к форме (количество примеров, количество признаков)
X_lab_pred = scaler_lab.inverse_transform(X_lab_pred_norm.reshape(-1, X_lab.shape[1]))

# mae_bridge показывает, насколько точно модель переводит признаки Audi в признаки Panasonic
mae_bridge = np.mean(np.abs(X_lab_pred - X_lab_test))
print(f"MAE моста: {mae_bridge:.4f}")

print("Применим метод моста к BMW i3")
# Шаг 1: Нормализуем ВСЕ признаки BMW i3 (X_target_norm) с помощью scaler_field
# scaler_field был обучен на X_field_train (признаках BMW i3 из пар)
# transform - только преобразование (без переобучения)
X_target_field_norm = scaler_field.transform(X_target_norm)

# X_target_norm - (8725, 21) - все нормализованные признаки BMW i3
# X_target_field_norm - (8725, 21) - те же данные, но приведенные к масштабу,
# который видел мост при обучении (среднее=0, std=1 на тренировочных парах)

# Шаг 2: Предсказываем "лабораторные" признаки Panasonic по признакам BMW i3
# bridge - обученная модель Random Forest
# На вход: нормализованные признаки BMW i3
# На выход: нормализованные признаки Panasonic (в том же масштабе, что и X_lab_norm)
X_target_lab_norm = bridge.predict(X_target_field_norm)
# X_target_lab_norm - (8725, 21) - предсказанные признаки в нормализованном виде
# Это то, как выглядели бы признаки BMW i3, если бы их измеряли в лаборатории Panasonic

# Шаг 3: Возвращаем предсказанные признаки в исходный масштаб Panasonic
# scaler_lab был обучен на X_lab_train (реальных признаках Panasonic из пар)
# inverse_transform - обратное преобразование: из нормализованного вида в исходный масштаб
# reshape(-1, X_lab.shape[1]) - приводим к форме (количество примеров, 21 признак)
X_target_lab = scaler_lab.inverse_transform(X_target_lab_norm.reshape(-1, X_lab.shape[1]))

# X_target_lab - (8725, 21) - финальные "лабораторные" признаки
# Они имеют тот же масштаб, что и исходные признаки Panasonic

print(f"После моста: {X_target_lab.shape}")

print("Обучение финальной модели")
X_src_train, X_src_test, y_src_train, y_src_test = train_test_split(
    X_source_norm, y_source, test_size=0.2, random_state=42)

final_model = RandomForestRegressor(n_estimators=100,
                                    max_depth=10,
                                    random_state=42,
                                    n_jobs=-1)
final_model.fit(X_src_train, y_src_train)

y_src_pred = final_model.predict(X_src_test)
src_mae = np.mean(np.abs(y_src_pred - y_src_test)) * 100
src_r2 = r2_score(y_src_test, y_src_pred)
print(f"LG HG2 test: MAE={src_mae:.2f}%, R²={src_r2:.4f}")

# 7. тестирование на BMWi3
print("тестирование на BMWi3")
y_tgt_pred = final_model.predict(X_target_lab)

mae = np.mean(np.abs(y_tgt_pred - y_target)) * 100
rmse = np.sqrt(np.mean((y_tgt_pred - y_target) ** 2)) * 100
r2 = r2_score(y_target, y_tgt_pred)
corr = np.corrcoef(y_target, y_tgt_pred)[0, 1]

print(f"\n Результаты на BMWi3:")
print(f"   MAE:  {mae:.2f}%")
print(f"   RMSE: {rmse:.2f}%")
print(f"   R²:   {r2:.4f}")
print(f"   Корреляция: {corr:.4f}")

# MAE по диапазонам
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

print(f"\n MAE по диапазонам:")
for i in range(len(bins) - 1):
    mask = (y_target >= bins[i]) & (y_target < bins[i + 1])
    if mask.sum() > 0:
        mae_range = np.mean(np.abs(y_tgt_pred[mask] - y_target[mask])) * 100
        print(f"   {bin_labels[i]}: {mae_range:.2f}% ({mask.sum()} окон)")

# 8. визуализация
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_src_test * 100, y_src_pred * 100, alpha=0.5, s=10)
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Идеал')
plt.xlabel('Реальный SoC (%)')
plt.ylabel('Предсказанный SoC (%)')
plt.title(f'Panasonic (test)\nMAE={src_mae:.2f}%')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_target * 100, y_tgt_pred * 100, alpha=0.5, s=10, c='green')
plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Идеал')
plt.xlabel('Реальный SoC (%)')
plt.ylabel('Предсказанный SoC (%)')
plt.title(f'BMW i3 (после моста)\nMAE={mae:.2f}%, R²={r2:.3f}')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('bridge_lg_hg2_to_bmw_results_панасоник.png', dpi=150)
plt.show()

print(f"\n ИТОГ: MAE на BMW i3 = {mae:.2f}%")