import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings('ignore')

# КОНФИГУРАЦИЯ
SOURCE_PATH = "/Users/nierra/Desktop/диплом-2/реальные данные/пара 2"
LG_HG2_PRE = os.path.join(SOURCE_PATH, "Panasonic_Preprocessed_final")
BMW_PRE = os.path.join(SOURCE_PATH, "BMW_Preprocessed_final")

# Параметры модели
WINDOW_SIZE = 100
STRIDE = 20
BATCH_SIZE = 128
HIDDEN_SIZE = 64
DEVICE = torch.device('cpu')

# Режим работы: 'train' или 'test'
MODE = 'train'

# Балансировка данных
MAX_SOURCE_SAMPLES = 50000   # Максимум окон из source (LG HG2)
MAX_TARGET_SAMPLES = 25000   # Целевое количество окон target после аугментации
TARGET_WEIGHT = 10.0         # Вес target в loss

print(f"Устройство: {DEVICE}")
print(f"Режим: {MODE}")
print(f"Window size: {WINDOW_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Source samples: {MAX_SOURCE_SAMPLES}")
print(f"Target samples: {MAX_TARGET_SAMPLES}")
print(f"Target weight: {TARGET_WEIGHT}")

# КЛАССЫ DANN
class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class OptimizedDANN(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super(OptimizedDANN, self).__init__()
        # Двунаправленный LSTM для извлечения признаков
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Агрегатор признаков
        self.feature_aggregator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Предсказатель SoC (регрессор)
        self.label_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Классификатор домена (DANN)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x, alpha=0.0):
        lstm_out, _ = self.lstm(x)
        features = torch.mean(lstm_out, dim=1)
        features = self.feature_aggregator(features)
        soc_pred = self.label_predictor(features)
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)
        return soc_pred, domain_pred

def load_lg_hg2_data(max_samples=None):
    if not os.path.exists(LG_HG2_PRE):
        print(f" Папка не найдена: {LG_HG2_PRE}")
        return np.array([]), np.array([])

    # Находим все папки с температурами
    temp_folders = [f for f in os.listdir(LG_HG2_PRE) if os.path.isdir(os.path.join(LG_HG2_PRE, f))]
    print(f"\n Загрузка panasonic (source)...")
    print(f"   Температур: {temp_folders}")

    all_windows = []
    all_targets = []
    all_masks = []
    windows_created = 0

    for temp_folder in tqdm(temp_folders, desc="Обработка"):
        folder_path = os.path.join(LG_HG2_PRE, temp_folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))

        # Исключаем зарядные и паузы
        files = [f for f in files if 'Charge' not in f and 'Paus' not in f]

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                # Колонки после финальной обработки
                features = df[['Current_A', 'Voltage_V', 'Temperature_C']].values.astype(np.float32)
                # SoC: переводим проценты в доли
                soc_raw = df['SOC_percent'].values.astype(np.float32)
                targets = soc_raw / 100.0

                # Маска (все 1, так как лабораторные данные)
                # В лабораторных данных LG HG2 SOC рассчитан кулоновским счетом и доступен в каждой точке.
                # Нет пропусков или "замороженных" значений, как в реальных данных Audi.
                masks = np.ones(len(df), dtype=bool)

                # Нарезаем длинный временной ряд на перекрывающиеся окна.
                for i in range(0, len(features) - WINDOW_SIZE, STRIDE):
                    if max_samples and windows_created >= max_samples:
                        break
                    window = features[i:i + WINDOW_SIZE]
                    target = targets[i + WINDOW_SIZE - 1]
                    mask = masks[i + WINDOW_SIZE - 1]

                    # Пропускает окна, содержащие NaN (пропуски) или бесконечные значения.
                    if np.isnan(window).any() or np.isinf(window).any() or np.isnan(target) or np.isinf(target):
                        continue

                    all_windows.append(window)
                    all_targets.append(target)
                    all_masks.append(mask)
                    windows_created += 1

            except Exception as e:
                continue

            if max_samples and windows_created >= max_samples:
                break

        if max_samples and windows_created >= max_samples:
            break
    print(f" Окон: {len(all_windows)}")
    print(f" Маска=1: {np.sum(all_masks)}/{len(all_masks)} ({np.sum(all_masks)/len(all_masks)*100:.1f}%)")
    print(f" SoC: {np.min(all_targets)*100:.1f}% - {np.max(all_targets)*100:.1f}%")

    return np.array(all_windows), np.array(all_targets), np.array(all_masks, dtype=bool)

def load_bmw_data():
    if not os.path.exists(BMW_PRE):
        print(f" Папка не найдена: {BMW_PRE}")
        return np.array([]), np.array([]), np.array([])

    files = sorted(glob.glob(os.path.join(BMW_PRE, "Trip*.csv")))

    all_windows = []
    all_targets = []
    all_masks = []
    windows_created = 0

    for file_path in tqdm(files, desc="Загрузка BMW i3"):
        try:
            df = pd.read_csv(file_path)
            features = df[['Current_A', 'Voltage_V', 'Temperature_C']].values.astype(np.float32)
            targets = df['SOC_percent'].values.astype(np.float32) / 100.0
            masks = df['Mask'].values.astype(bool) if 'Mask' in df.columns else np.ones(len(df), dtype=bool)

            for i in range(0, len(features) - WINDOW_SIZE, STRIDE):
                # Используем только достоверные метки (Mask=1)
                if not masks[i + WINDOW_SIZE - 1]:
                    continue

                window = features[i:i + WINDOW_SIZE]
                target = targets[i + WINDOW_SIZE - 1]
                mask = masks[i + WINDOW_SIZE - 1]

                if np.isnan(window).any() or np.isinf(window).any() or np.isnan(target) or np.isinf(target):
                    continue

                all_windows.append(window)
                all_targets.append(target)
                all_masks.append(mask)
                windows_created += 1

        except Exception as e:
            continue

    print(f" Окон: {len(all_windows)}")
    print(f" Маска=1: {np.sum(all_masks)}/{len(all_masks)} ({np.sum(all_masks)/len(all_masks)*100:.1f}%)")
    print(f" SoC: {np.min(all_targets)*100:.1f}% - {np.max(all_targets)*100:.1f}%")

    return np.array(all_windows), np.array(all_targets), np.array(all_masks, dtype=bool)

# Аугментация target данных для балансировки
def augment_target_data(X, y, masks, target_count=25000):
    mask_indices = np.where(masks)[0]
    n_masked = len(mask_indices)

    if n_masked == 0:
        return X, y, masks

    print(f" Исходных target с маской: {n_masked}")

    X_aug = list(X)
    y_aug = list(y)
    masks_aug = list(masks)

    n_to_generate = min(target_count - n_masked, n_masked * 5)

    if n_to_generate > 0:
        for _ in range(n_to_generate):
            idx = np.random.choice(mask_indices)
            noise = np.random.normal(0, 0.01, X[idx].shape)
            X_aug.append(X[idx] + noise)
            y_aug.append(y[idx])
            masks_aug.append(True)

    print(f" После аугментации: {len(masks_aug)} окон, маска=1: {sum(masks_aug)}")

    return np.array(X_aug), np.array(y_aug), np.array(masks_aug)

# Загружает сбалансированные данных
def load_balanced_data():
    total_start = time.time()
    # Загружаем LG HG2 (source)
    print(f"\n Загрузка LG HG2 (max {MAX_SOURCE_SAMPLES} окон)...")
    X_source, y_source, mask_source = load_lg_hg2_data(max_samples=MAX_SOURCE_SAMPLES)

    # Загружаем BMW i3 (target)
    print(f"\n Загрузка BMW i3...")
    X_target, y_target, mask_target = load_bmw_data()

    # Аугментация target
    print(f"\n Аугментация target данных...")
    X_target, y_target, mask_target = augment_target_data(X_target, y_target, mask_target,target_count=MAX_TARGET_SAMPLES)
    print(f"\n Всего загружено за {time.time() - total_start:.1f} сек")

    return X_source, y_source, mask_source, X_target, y_target, mask_target

# датасет с весами
class BalancedDataset(Dataset):
    def __init__(self, X, y, domains, masks):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        self.domains = torch.LongTensor(domains)
        self.masks = torch.BoolTensor(masks)
        self.weights = self._compute_weights()

    def _compute_weights(self):
        weights = []
        for i in range(len(self.X)):
            if self.domains[i] == 1 and self.masks[i]:
                weights.append(TARGET_WEIGHT)
            else:
                weights.append(1.0)
        return torch.DoubleTensor(weights)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks[idx], self.domains[idx]

# обучение DANN
def train_balanced_dann(model, train_loader, val_loader, epochs=25, lambda_domain=0.05):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    soc_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    print(f"\n Обучение DANN (с балансировкой)")

    best_target_mae = float('inf')
    history = {'soc_loss': [], 'domain_loss': [], 'val_mae': []}

    for epoch in range(epochs):
        model.train()
        total_soc_loss = 0
        total_domain_loss = 0
        batch_count = 0
        target_hits = 0

        for X, y, masks, domains in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            masks = masks.to(DEVICE)
            domains = domains.to(DEVICE)

            alpha = lambda_domain * (epoch / epochs)
            soc_pred, domain_pred = model(X, alpha)

            source_mask = (domains == 0) & masks
            target_mask = (domains == 1) & masks
            target_hits += target_mask.sum().item()

            loss_source = soc_criterion(soc_pred[source_mask], y[source_mask]) if source_mask.sum() > 0 else 0
            loss_target = soc_criterion(soc_pred[target_mask], y[target_mask]) if target_mask.sum() > 0 else 0

            if target_mask.sum() == 0:
                soc_loss = loss_source
            else:
                soc_loss = loss_source + TARGET_WEIGHT * loss_target

            domain_loss = domain_criterion(domain_pred, domains)
            loss = soc_loss + alpha * domain_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_soc_loss += soc_loss.item() if isinstance(soc_loss, torch.Tensor) else 0
            total_domain_loss += domain_loss.item()
            batch_count += 1

        scheduler.step()
        avg_soc_loss = total_soc_loss / batch_count if batch_count > 0 else 0
        avg_domain_loss = total_domain_loss / batch_count if batch_count > 0 else 0
        history['soc_loss'].append(avg_soc_loss)
        history['domain_loss'].append(avg_domain_loss)

        val_results = validate_model(model, val_loader)
        history['val_mae'].append(val_results['mae'])

        if val_results['mae'] < best_target_mae:
            best_target_mae = val_results['mae']
            torch.save(model.state_dict(), 'best_dann_lg_hg2_to_bmw.pth')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | SOC Loss: {avg_soc_loss:.4f} | "
                  f"Domain Loss: {avg_domain_loss:.4f} | α: {alpha:.3f} | "
                  f"Target MAE: {val_results['mae']:.2f}% | Target hits: {target_hits}")

    print(f"\n DANN обучен, лучший MAE: {best_target_mae:.2f}%")
    return history, best_target_mae


def validate_model(model, val_loader):
    model.eval()
    target_mae = 0
    target_count = 0

    with torch.no_grad():
        for X, y, masks, domains in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            masks = masks.to(DEVICE)
            soc_pred, _ = model(X, alpha=0)

            target_mask = (domains == 1) & masks.cpu()
            if target_mask.sum() > 0:
                preds = soc_pred[target_mask].cpu().numpy().flatten()
                targets = y[target_mask].cpu().numpy().flatten()
                target_mae += np.sum(np.abs(preds - targets))
                target_count += len(preds)

    if target_count == 0:
        return {'mae': 0}

    return {'mae': (target_mae / target_count) * 100}

# тестирование
def test_model_detailed(model, test_loader, name="Model"):
    """Детальное тестирование модели"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, masks, domains in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            masks = masks.to(DEVICE)
            soc_pred, _ = model(X, alpha=0)

            target_mask = (domains == 1) & masks.cpu()
            if target_mask.sum() > 0:
                all_preds.extend(soc_pred[target_mask].cpu().numpy().flatten())
                all_targets.extend(y[target_mask].cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    if len(all_preds) == 0:
        return {'mae': 0, 'r2': 0, 'correlation': 0, 'preds': all_preds, 'targets': all_targets}

    mae = np.mean(np.abs(all_preds - all_targets)) * 100
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2)) * 100
    correlation = np.corrcoef(all_targets, all_preds)[0, 1]

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    mean_pred = np.mean(all_preds) * 100
    mean_target = np.mean(all_targets) * 100
    std_pred = np.std(all_preds) * 100
    std_true = np.std(all_targets) * 100
    max_error = np.max(np.abs(all_preds - all_targets)) * 100
    within_10 = np.mean(np.abs(all_preds - all_targets) < 0.1) * 100

    print(f"\n{'='*60}")
    print(f" {name} - ДЕТАЛЬНАЯ ДИАГНОСТИКА")
    print(f"{'='*60}")
    print(f"   MAE:            {mae:.2f}%")
    print(f"   RMSE:           {rmse:.2f}%")
    print(f"   R²:             {r2:.4f}")
    print(f"   Корреляция:     {correlation:.4f}")
    print(f"   Среднее (pred): {mean_pred:.2f}%")
    print(f"   Среднее (true): {mean_target:.2f}%")
    print(f"   Std (pred):     {std_pred:.2f}%")
    print(f"   Std (true):     {std_true:.2f}%")
    print(f"   Max error:      {max_error:.2f}%")
    print(f"   Within 10%:     {within_10:.1f}%")
    print(f"   Тестовых окон:  {len(all_preds)}")

    # MAE по диапазонам
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    print(f"\n📊 MAE по диапазонам:")
    for i in range(len(bins) - 1):
        mask = (all_targets >= bins[i]) & (all_targets < bins[i + 1])
        if mask.sum() > 0:
            mae_range = np.mean(np.abs(all_preds[mask] - all_targets[mask])) * 100
            print(f"   {bin_labels[i]}: {mae_range:.2f}% ({mask.sum()} окон)")

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'correlation': correlation,
        'mean_pred': mean_pred, 'mean_target': mean_target,
        'std_pred': std_pred, 'std_true': std_true,
        'max_error': max_error, 'within_10': within_10,
        'preds': all_preds, 'targets': all_targets
    }


def main():
    # 1. Загружаем сбалансированные данные
    X_source, y_source, mask_source, X_target, y_target, mask_target = load_balanced_data()

    # 2. Нормализация
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

    # 3. Объединяем данные для обучения
    X_all = np.vstack([X_source_norm, X_target_norm])
    y_all = np.hstack([y_source, y_target])
    domains_all = np.hstack([np.zeros(len(X_source)), np.ones(len(X_target))])
    masks_all = np.hstack([mask_source, mask_target])

    print(f"\n ИТОГОВЫЙ ДАТАСЕТ:")
    print(f"   Всего окон: {len(X_all)}")
    print(f"   Source (LG HG2): {len(X_source)}")
    print(f"   Target (BMW i3): {len(X_target)}")
    print(f"   Target маска=1: {mask_target.sum()}/{len(mask_target)} ({mask_target.sum()/len(mask_target)*100:.1f}%)")

    # 4. Разделение на train/val/test
    indices = np.random.permutation(len(X_all))
    train_split = int(0.7 * len(X_all))
    val_split = int(0.85 * len(X_all))

    train_dataset = BalancedDataset(
        X_all[indices[:train_split]], y_all[indices[:train_split]],
        domains_all[indices[:train_split]], masks_all[indices[:train_split]]
    )
    val_dataset = BalancedDataset(
        X_all[indices[train_split:val_split]], y_all[indices[train_split:val_split]],
        domains_all[indices[train_split:val_split]], masks_all[indices[train_split:val_split]]
    )
    test_dataset = BalancedDataset(
        X_all[indices[val_split:]], y_all[indices[val_split:]],
        domains_all[indices[val_split:]], masks_all[indices[val_split:]]
    )

    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n Размеры:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val:   {len(val_dataset)}")
    print(f"   Test:  {len(test_dataset)}")

    # 5. Создаём модель
    dann = OptimizedDANN(hidden_size=HIDDEN_SIZE)

    # 6. Обучаем
    dann_history, dann_best = train_balanced_dann(dann, train_loader, val_loader)

    # 7. Тестируем
    dann_results = test_model_detailed(dann, test_loader, "DANN (LG HG2 → BMW i3)")

    # 8. Визуализация
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(dann_results['targets'] * 100, dann_results['preds'] * 100,
                alpha=0.5, s=10, c='red')
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Идеал')
    plt.xlabel('Реальный SoC (%)')
    plt.ylabel('Предсказанный SoC (%)')
    plt.title(f'DANN (панасоник → BMW i3)\nMAE={dann_results["mae"]:.2f}%, R²={dann_results["r2"]:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dann_history['val_mae'], 'r-o', label='DANN')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('dann_панасоник_to_bmw.png', dpi=150)
    plt.show()

    print(f"\n Результаты сохранены в 'dann_панасоник_to_bmw.png'")
    print(f"\n ИТОГ: MAE на BMW i3 = {dann_results['mae']:.2f}%")
if __name__ == "__main__":
    main()