import numpy as np
import json
import random, os
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from typing import Tuple, Literal, Union, Dict, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from utils import QUALITY


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(42)


def process_video_data(json_data: Dict, val_videos: List[str], train_videos: List[str], gt_values: List[int],
                      window_size: int = 8, overlap_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def trim_zeros_and_process(video_data: Dict) -> Dict:
        """Пункт 1: Удаление нулей с краев welding_zone_temp"""
        welding_temp = np.array(video_data['welding_zone_temp'])
        
        # Находим индексы ненулевых элементов
        non_zero_indices = np.where(welding_temp != 0)[0]
        
        if len(non_zero_indices) == 0:
            # Если все значения нулевые, возвращаем пустые массивы
            return {key: np.array([]) for key in video_data.keys()}
        
        start_idx = non_zero_indices[0]
        end_idx = non_zero_indices[-1] + 1
        
        # Обрезаем все массивы
        trimmed_data = {}
        for key, values in video_data.items():
            trimmed_data[key] = np.array(values)[start_idx:end_idx]
            
        return trimmed_data
    
    def split_into_quarters(data: np.ndarray) -> List[np.ndarray]:
        """Пункт 2: Разделение на 4 равные части"""
        if len(data) == 0:
            return [np.array([]) for _ in range(4)]
        
        part_length = len(data) // 4
        parts = []
        
        for i in range(4):
            start_idx = i * part_length
            if i == 3:  # Последняя часть - все оставшиеся элементы
                end_idx = len(data)
            else:
                end_idx = (i + 1) * part_length
            parts.append(data[start_idx:end_idx])
            
        return parts
    
    def calculate_moving_stats(data: np.ndarray, window_size: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Вычисление скользящего среднего и СКО с заданным шагом"""
        if len(data) < window_size:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        means = []
        stds = []
        medians = []
        rms = []  # среднеквадратичное значение
        slopes = []  # наклон тренда
        
        for i in range(0, len(data) - window_size + 1, step):
            window = data[i:i + window_size]
            
            means.append(np.mean(window))
            stds.append(np.std(window))
            medians.append(np.median(window))
            rms.append(np.sqrt(np.mean(window**2)))
            
            # Вычисляем наклон линейного тренда
            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0] if len(window) > 1 else 0
            slopes.append(slope)
            
        return np.array(means), np.array(stds), np.array(medians), np.array(rms), np.array(slopes)
    
    def calculate_step_size(window_size: int, overlap_ratio: float) -> int:
        """Вычисление шага для скользящего окна на основе доли пересечения"""
        if overlap_ratio <= 0:
            return window_size  # Без пересечения
        elif overlap_ratio >= 1:
            return 1  # Максимальное пересечение (шаг 1)
        else:
            # Вычисляем шаг: window_size * (1 - overlap_ratio)
            step = int(window_size * (1 - overlap_ratio))
            return max(1, step)  # Шаг не меньше 1
        
    def contains_zeros(window_data: np.ndarray) -> bool:
        """Проверяет, содержит ли окно нулевые значения"""
        return np.any(window_data == 0)
    
    processed_data = {}
    
    # Вычисляем шаг для окон
    step_size = calculate_step_size(window_size, overlap_ratio)
    
    for video_name, video_data in json_data.items():
        # 1. Удаляем нули с краев
        trimmed_data = trim_zeros_and_process(video_data)
        
        if len(trimmed_data['welding_zone_temp']) == 0:
            continue
            
        # 2. Разделяем на 4 части и получаем GT значения для каждой четверти
        quarters_data = {}
        
        # Разделяем все признаки на четверти
        for key in ['size', 'n_spatters', 'temp', 'welding_zone_temp']:
            quarters = split_into_quarters(trimmed_data[key])
            quarters_data[key] = quarters
        
        # Получаем GT значения для каждой четверти (предполагаем, что они уже есть в данных)
        # Если GT передаются отдельно, нужно изменить эту часть
        # gt_quarters = split_into_quarters(trimmed_data['welding_zone_temp'])
        # quarter_gt_values = [np.mean(q) if len(q) > 0 else 0 for q in gt_quarters]
        
        # 3. Вычисляем скользящие статистики для каждой четверти
        features = []
        gt_for_features = []
        
        for quarter_idx in range(4):
            # Получаем данные для текущей четверти
            quarter_data_size = quarters_data['size'][quarter_idx]
            quarter_data_spatters = quarters_data['n_spatters'][quarter_idx]
            quarter_data_temp = quarters_data['temp'][quarter_idx]
            quarter_data_welding = quarters_data['welding_zone_temp'][quarter_idx]
            
            # GT для всей этой четверти
            quarter_gt = gt_values[video_name][quarter_idx]
            
            if len(quarter_data_size) >= window_size:
                size_means, size_stds, size_medians, size_rms, size_slopes = calculate_moving_stats(quarter_data_size, window_size, step_size)
                spatter_means, spatter_stds, spatter_medians, spatter_rms, spatter_slopes = calculate_moving_stats(quarter_data_spatters, window_size, step_size)
                temp_means, temp_stds, temp_medians, temp_rms, temp_slopes = calculate_moving_stats(quarter_data_temp, window_size, step_size)
                welding_means, welding_stds, welding_medians, welding_rms, welding_slopes = calculate_moving_stats(quarter_data_welding, window_size, step_size)

                # Для каждого окна создаем вектор признаков
                for i in range(len(size_means)):
                    window_features = {
                        # Основные статистики (работали хорошо)
                        'size_mean': size_means[i],
                        'size_std': size_stds[i],
                        'n_spatters_mean': spatter_means[i],
                        'n_spatters_std': spatter_stds[i],
                        'temp_mean': temp_means[i],
                        'temp_std': temp_stds[i],
                        'welding_zone_temp_mean': welding_means[i],
                        'welding_zone_temp_std': welding_stds[i],
                        
                        # Новые статистики
                        'size_median': size_medians[i],
                        'size_rms': size_rms[i],
                        'size_slope': size_slopes[i],
                        'n_spatters_median': spatter_medians[i],
                        'n_spatters_rms': spatter_rms[i],
                        'n_spatters_slope': spatter_slopes[i],
                        'temp_median': temp_medians[i],
                        'temp_rms': temp_rms[i],
                        'temp_slope': temp_slopes[i],
                        'welding_zone_temp_median': welding_medians[i],
                        'welding_zone_temp_rms': welding_rms[i],
                        'welding_zone_temp_slope': welding_slopes[i],
                        
                        'quarter': quarter_idx + 1
                    }
                    
                    features.append(window_features)
                    gt_for_features.append(quarter_gt)  # Всем окнам из этой четверти - одинаковый GT
        
        processed_data[video_name] = {
            'features': features,
            'gt': gt_for_features
        }
    
    # 4. Создаем датасеты
    def create_dataset(video_names: List[str]) -> pd.DataFrame:
        """Создание датасета для списка видео"""
        rows = []
        
        for video_name in video_names:
            if video_name not in processed_data:
                continue
                
            video_info = processed_data[video_name]
            
            # Создаем строки для каждого окна
            for i, feature_set in enumerate(video_info['features']):
                row = feature_set.copy()
                row['video_name'] = video_name
                row['gt'] = video_info['gt'][i]  # GT для этого окна (одинаковый для всей четверти)
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    # Создаем тренировочный и валидационный датасеты
    train_df = create_dataset(train_videos)
    val_df = create_dataset(val_videos)
    
    return train_df, val_df

videos = list(QUALITY.keys())

with open('thermograms_analysis/metrics/without_tracker.json', 'r') as f:
    data = json.load(f)

labels = {}
gt = {vid: [] for vid in videos}
cl = 6  # 1 - отл, 3 - не оч, 4 - отл, 5 - отл, 6 - хор
for vid in videos:
    label = 0
    for d in QUALITY[vid]:
        if d[6] > 0:
            label = 1
        gt[vid].append(int(d[cl] > 0))

    labels[vid] = label

val_videos = ['thermogram_16.npy', 'thermogram_20.npy', 'thermogram_14.npy', 'thermogram_23.npy', 'thermogram_25.npy', 'thermogram_11.npy', 'thermogram_31.npy', 'thermogram_17.npy']
train_videos = [vid for vid in videos if vid not in val_videos]

# train_df, val_df = process_video_data(data, val_videos, train_videos, gt, window_size=16, overlap_ratio=0.5)
# y_train = train_df['gt']
# y_val = val_df['gt']
# train_df = train_df.drop(['gt', 'video_name', 'quarter'], axis=1)
# val_df = val_df.drop(['gt', 'video_name', 'quarter'], axis=1)

# print(train_df.head())
# print(train_df.shape, val_df.shape)

# # model = CatBoostClassifier(logging_level='Silent')
# # model = RandomForestClassifier()
# # model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
# # model = make_pipeline(StandardScaler(), SVC(probability=True))
# model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,8), max_iter=200, activation='relu'))

# model.fit(train_df, y_train)
# pred_proba = model.predict_proba(val_df)[:, 1]
# precision, recall, _ = precision_recall_curve(y_val, pred_proba)
# print(f"AP: {average_precision_score(y_val, pred_proba)}")
# precision = precision.tolist()
# precision.append(1)
# precision.insert(0, 0)
# recall = recall.tolist()
# recall.append(0)
# recall.insert(0, 1)
# plt.plot(recall, precision, lw=2, color='black')
# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.show()


folds = 4
k_fold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
output = {}
y_real = []
y_proba = []
for i, (train_index, val_index) in tqdm(enumerate(k_fold.split(list(labels.keys()), list(labels.values()))), total=folds):
    train_videos = [list(labels.keys())[v] for v in train_index]
    val_videos = [list(labels.keys())[v] for v in val_index]
    print(val_videos)
    train_df, val_df = process_video_data(data, val_videos, train_videos, gt, window_size=16, overlap_ratio=0.5)
    y_train = train_df['gt']
    y_val = val_df['gt']
    train_df = train_df.drop(['gt', 'video_name', 'quarter'], axis=1)
    val_df = val_df.drop(['gt', 'video_name', 'quarter'], axis=1)
    model = CatBoostClassifier(logging_level='Silent')
    # model = RandomForestClassifier()
    # model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LogisticRegression())
    # model = make_pipeline(StandardScaler(), SVC(probability=True))
    # model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,8), max_iter=200, activation='relu'))
    model.fit(train_df, y_train)
    pred_proba = model.predict_proba(val_df)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, pred_proba)
    lab = 'Fold %d AP=%.3f' % (i+1, average_precision_score(y_val, pred_proba))
    precision = precision.tolist()
    precision.append(1)
    precision.insert(0, 0)
    recall = recall.tolist()
    recall.append(0)
    recall.insert(0, 1)

    output[lab] = [precision, recall]
    y_real.append(y_val)
    plt.plot(recall, precision, label=lab)
    y_proba.append(pred_proba)

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba)
lab = 'Overall AP=%.3f' % (average_precision_score(y_real, y_proba))
precision = precision.tolist()
precision.append(1)
precision.insert(0, 0)
recall = recall.tolist()
recall.append(0)
recall.insert(0, 1)
output[lab] = [precision, recall]
plt.plot(recall, precision, label=lab, lw=2, color='black')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend(loc='lower left', fontsize='small')
plt.show()