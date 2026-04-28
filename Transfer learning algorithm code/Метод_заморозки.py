import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2"
LG_HG2_PRE = os.path.join(SOURCE_PATH, "Panasonic_Preprocessed_final")
BMW_PRE = os.path.join(SOURCE_PATH, "BMW_Preprocessed_final")

# Параметры модели
WINDOW_SIZE = 100
STRIDE = 20
BATCH_SIZE = 128
HIDDEN_SIZE = 64
EPOCHS_PHASE1 = 20  # Фаза 1: только регрессор (LSTM заморожен)
EPOCHS_PHASE2 = 10  # Фаза 2: дообучение всех слоёв
MAX_SOURCE = 50000  # Максимум окон из source (LG HG2)
TARGET_AUGMENT = 5  # Аугментация target (BMW)

DEVICE = torch.device('cpu')
print(" LSTM С ЗАМОРОЗКОЙ - LG HG2 → BMW i3")

# функция загрузки LG HG2 (SOURCE)
def load_lg_hg2_data(max_samples=None):
    if not os.path.exists(LG_HG2_PRE):
        print(f" Папка не найдена: {LG_HG2_PRE}")
        return np.array([]), np.array([])

    # Находим все папки с температурами
    temp_folders = [f for f in os.listdir(LG_HG2_PRE) if os.path.isdir(os.path.join(LG_HG2_PRE, f))]
    print(f"\n Загрузка panasonic (source)...")
    print(f"   Температур: {temp_folders}")

    X, y = [], []
    count = 0

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
                tgt = df['SOC_percent'].values.astype(np.float32)/ 100.0

                # Нарезаем на окна
                for i in range(0, len(feat) - WINDOW_SIZE, STRIDE):
                    if max_samples and count >= max_samples:
                        break

                    window = feat[i:i + WINDOW_SIZE]
                    target = tgt[i + WINDOW_SIZE - 1]

                    if np.isnan(window).any() or np.isinf(window).any() or np.isnan(target) or np.isinf(target):
                        continue

                    # Извлекаем 21 признак
                    X.append(window)
                    y.append(target)
                    count += 1

            except Exception as e:
                continue

            if max_samples and count >= max_samples:
                break

        if max_samples and count >= max_samples:
            break

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
        msk = df['Mask'].values.astype(bool) if 'Mask' in df.columns else np.ones(len(df), dtype=bool)

        for i in range(0, len(feat) - WINDOW_SIZE, STRIDE):
            if max_samples and count >= max_samples:
                break

            if not msk[i + WINDOW_SIZE - 1]:
                continue

            window = feat[i:i + WINDOW_SIZE]
            target = tgt[i + WINDOW_SIZE - 1]
            mask = msk[i + WINDOW_SIZE - 1]

            if np.isnan(window).any() or np.isinf(window).any() or np.isnan(target) or np.isinf(target):
                continue

            X.append(window)
            y.append(target)
            masks.append(mask)
            count += 1

            # Аугментация для target
            if augment_factor > 1:
                for _ in range(augment_factor - 1):
                    noise = np.random.normal(0, 0.01, window.shape)
                    X.append(window + noise)
                    y.append(target)
                    masks.append(1)
                    count += 1

        if max_samples and count >= max_samples:
            break

    return np.array(X), np.array(y), np.array(masks, dtype=bool)

# КЛАСС МОДЕЛИ
class FreezeLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(FreezeLSTM, self).__init__()
        # двунаправленная нейросеть
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        features = torch.mean(lstm_out, dim=1)
        return self.regressor(features)

    def freeze_layers(self, freeze=True):
        for param in self.lstm.parameters():
            param.requires_grad = not freeze

# Данные X преобразуются в тензор PyTorch
# типа FloatTensor (32-битные числа с плавающей запятой). Это стандартный тип данных для входных признаков в PyTorch.
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, masks):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.masks = torch.BoolTensor(masks)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.masks[idx]

def train_phase(model, loader, val_loader, epochs, freeze=True, lr=0.001):
    if freeze:
        model.freeze_layers(True)
        print(f"\nФаза 1: Заморожен LSTM")
    else:
        model.freeze_layers(False)
        print(f"\nФаза 2: Дообучение всех слоев")

    # перенос модели на CPU
    model = model.to(DEVICE)
    # Включаем в оптимизатор только обучаемые параметры
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'val_mae': []}
    best_mae = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss, batch_count = 0, 0

        for X, y, masks in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            masks = masks.to(DEVICE)
            # батчи без валидных данных игнорим
            if masks.sum() == 0:
                continue

            optimizer.zero_grad()
            pred = model(X)
            # вычисляется ошибка только на валидных позициях (где masks=True)
            loss = criterion(pred[masks], y[masks])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        # Валидация
        model.eval()
        val_mae, val_count = 0, 0
        # отключили градиенты для ускорения экономит память и ускоряет вычисления,
        # так как для валидации обратное распространение не нужно
        with torch.no_grad():
            # по всем батчам идем
            for X, y, masks in val_loader:
                # на cpu
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                valid = masks.bool()
                # есть ли хотябы одна точка с маской в батче
                if valid.sum() > 0:
                    # Суммирует абсолютные ошибки только для точек с маской и считаем их количество
                    val_mae += torch.abs(pred[valid] - y[valid]).sum().item()
                    val_count += valid.sum().item()

        # средняя абсолютная ошибка в процентах
        val_mae = (val_mae / val_count) * 100 if val_count > 0 else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        history['loss'].append(avg_loss)
        history['val_mae'].append(val_mae)

        if val_mae < best_mae:
            best_mae = val_mae

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch + 1:2d}/{epochs} | Loss: {avg_loss:.6f} | Val MAE: {val_mae:.2f}%")

    return history, best_mae

def test_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y, masks in loader:
            X = X.to(DEVICE)
            pred = model(X)
            valid = masks.bool()
            if valid.sum() > 0:
                # Извлекаем предсказания для валидных точек, переводим в numpy и добавляем в список
                preds.extend(pred[valid].cpu().numpy().flatten())
                # Извлекаем реальные значения для валидных точек и добавляем в список
                targets.extend(y[valid].cpu().numpy().flatten())

    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets)) * 100
    rmse = np.sqrt(np.mean((preds - targets) ** 2)) * 100
    r2 = r2_score(targets, preds)
    corr = np.corrcoef(targets, preds)[0, 1]

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'corr': corr, 'preds': preds, 'targets': targets}

def main():
    print("Загрузка данных")
    # Загружаем source (LG HG2)
    X_source, y_source = load_lg_hg2_data(max_samples=MAX_SOURCE)
    # Загружаем target (BMW i3)
    X_target, y_target, masks_target = load_bmw_data(augment_factor=TARGET_AUGMENT)

    print(f"\n Данные загружены:")
    print(f"   Source (LG HG2): {X_source.shape}")
    print(f"   Target (Audi): {X_target.shape}")
    print(f"   Target с маской: {masks_target.sum()}")

    # Нормализуем source (LG HG2) отдельно
    scaler_source = StandardScaler()
    source_flat = X_source.reshape(-1, 3)
    X_source_norm = scaler_source.fit_transform(source_flat).reshape(X_source.shape)

    # Нормализуем target (BMW) отдельно
    scaler_target = StandardScaler()
    target_flat = X_target.reshape(-1, 3)
    X_target_norm = scaler_target.fit_transform(target_flat).reshape(X_target.shape)

    print(f"Source (LG HG2): mean={X_source_norm.mean():.3f}, std={X_source_norm.std():.3f}")
    print(f"Target (BMW i3): mean={X_target_norm.mean():.3f}, std={X_target_norm.std():.3f}")

    # 1. Разделяем source на train/val
    X_source_train, X_source_val, y_source_train, y_source_val = (
        train_test_split(X_source_norm, y_source, test_size=0.2, random_state=42))

    # 2. Разделяем target (Audi) на train/val/test
    X_target_train, X_target_temp, y_target_train, y_target_temp, masks_target_train, masks_target_temp = (
        train_test_split(X_target_norm, y_target, masks_target, test_size=0.4, random_state=42))

    X_target_val, X_target_test, y_target_val, y_target_test, masks_target_val, masks_target_test = (
        train_test_split(X_target_temp, y_target_temp, masks_target_temp, test_size=0.5, random_state=42))

    # 3. Объединяем для обучения
    X_train = np.vstack([X_source_train, X_target_train])
    y_train = np.hstack([y_source_train, y_target_train])
    masks_train = np.hstack([np.ones(len(X_source_train)), masks_target_train])

    # 4. Валидация (только target)
    X_val = X_target_val
    y_val = y_target_val
    masks_val = masks_target_val

    # 5. Тест (только target)
    X_test = X_target_test
    y_test = y_target_test
    masks_test = masks_target_test

    print(f"   Train: Source={len(X_source_train)}, Target={len(X_target_train)}")
    print(f"   Val:   Target={len(X_val)} (маска=1: {masks_val.sum()})")
    print(f"   Test:  Target={len(X_test)} (маска=1: {masks_test.sum()})")

    # Датасеты и загрузчики
    train_ds = TimeSeriesDataset(X_train, y_train, masks_train)
    val_ds = TimeSeriesDataset(X_val, y_val, masks_val)
    test_ds = TimeSeriesDataset(X_test, y_test, masks_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # model — модель FreezeLSTM
    # loader — обучающий DataLoader
    # val_loader — валидационный DataLoader
    # epochs — количество эпох обучения
    # freeze - определяем фазу
    model = FreezeLSTM(hidden_size=HIDDEN_SIZE)
    # Фаза 1: заморожен LSTM
    hist1, best1 = train_phase(model, train_loader, val_loader, EPOCHS_PHASE1, freeze=True, lr=0.001)
    # Фаза 2: дообучение всех слоёв
    hist2, best2 = train_phase(model, train_loader, val_loader, EPOCHS_PHASE2, freeze=False, lr=0.0005)
    print(f"\nЛучший MAE (фаза 1): {best1:.2f}%")
    print(f"  Лучший MAE (фаза 2): {best2:.2f}%")

    # тестирование
    results = test_model(model, test_loader)

    print(f"\n РЕЗУЛЬТАТЫ НА BMW i3:")
    print(f"   MAE:  {results['mae']:.2f}%")
    print(f"   RMSE: {results['rmse']:.2f}%")
    print(f"   R²:   {results['r2']:.4f}")
    print(f"   Корреляция: {results['corr']:.4f}")

    # 8. визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(results['targets'] * 100, results['preds'] * 100, alpha=0.5, s=10, c='blue')
    plt.plot([0, 100], [0, 100], 'r--', lw=2, label='Идеал')
    plt.xlabel('Реальный SoC (%)')
    plt.ylabel('Предсказанный SoC (%)')
    plt.title(f'BMW i3 (LSTM с заморозкой)\nMAE={results["mae"]:.2f}%, R²={results["r2"]:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    all_mae = hist1['val_mae'] + hist2['val_mae']
    plt.plot(all_mae, 'b-o', label='Val MAE')
    plt.axvline(x=len(hist1['val_mae']), color='r', linestyle='--', label='Начало фазы 2')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (%)')
    plt.title('Кривая обучения')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('freeze_lstm_панасоник_to_bmw.png', dpi=150)
    plt.show()

    print(f"\n ИТОГ: MAE на BMW i3 = {results['mae']:.2f}%")

if __name__ == "__main__":
    main()