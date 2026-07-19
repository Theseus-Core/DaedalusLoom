import os
import numpy as np

from config import LABEL_MAP, INV_LABEL_MAP_EN as INV_LABEL_MAP

def _load_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder directory '{folder_path}' not found.")
    
    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    X_dict = {}
    
    print(f"Loading data from {folder_path}...")
    for filename in npz_files:
        file_path = os.path.join(folder_path, filename)
        
        # Robust label extraction: find index before the 8-digit date string starting with '202'
        parts = filename.split('_')
        date_idx = -1
        for idx, part in enumerate(parts):
            if len(part) == 8 and part.isdigit() and part.startswith('202'):
                date_idx = idx
                break
        
        if date_idx != -1:
            label_name = '_'.join(parts[:date_idx])
        else:
            label_name = parts[0]
            
        if label_name not in LABEL_MAP:
            print(f"  Warning: Skipping unknown label prefix '{label_name}' in file '{filename}'")
            continue
            
        label_idx = LABEL_MAP[label_name]
        data = np.load(file_path, allow_pickle=True)['dataset']
        
        if label_idx not in X_dict:
            X_dict[label_idx] = []
        X_dict[label_idx].append(data)
        
    X_by_label = {}
    for idx in X_dict:
        X_by_label[idx] = np.concatenate(X_dict[idx], axis=0)
        print(f"  Class '{INV_LABEL_MAP.get(idx, str(idx))}': {X_by_label[idx].shape[0]} samples")
        
    return X_by_label

def load_dataset(dataset_dir="dataset/dataset_2026_7_19_2"):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
    
    # Check if 'train' and 'test' subdirectories exist
    train_path = os.path.join(dataset_dir, "train")
    test_path = os.path.join(dataset_dir, "test")
    
    if os.path.isdir(train_path) and os.path.isdir(test_path):
        print(f"Detected train/test split directories in {dataset_dir}")
        train_data = _load_folder(train_path)
        test_data = _load_folder(test_path)
        return {
            "train": train_data,
            "test": test_data
        }
    else:
        # Fallback to loading the directory directly (the old behavior)
        return _load_folder(dataset_dir)

def split_80_20(X_by_label):
    if isinstance(X_by_label, dict) and "train" in X_by_label and "test" in X_by_label:
        train_dict = X_by_label["train"]
        test_dict = X_by_label["test"]
        
        x_train_list, x_test_list = [], []
        y_train_list, y_test_list = [], []
        
        print("\nAssembling pre-split dataset...")
        # Merge all class indices from both train and test
        all_classes = sorted(list(set(train_dict.keys()) | set(test_dict.keys())))
        for idx in all_classes:
            train_samples = train_dict.get(idx, np.empty((0,)))
            test_samples = test_dict.get(idx, np.empty((0,)))
            
            if len(train_samples) > 0:
                x_train_list.append(train_samples)
                y_train_list.append(np.full(len(train_samples), idx, dtype=np.int64))
            if len(test_samples) > 0:
                x_test_list.append(test_samples)
                y_test_list.append(np.full(len(test_samples), idx, dtype=np.int64))
                
            print(f"  Class '{INV_LABEL_MAP.get(idx, str(idx))}': Train={len(train_samples)}, Test={len(test_samples)}")
            
        x_train = np.concatenate(x_train_list, axis=0) if x_train_list else np.empty((0,))
        x_test = np.concatenate(x_test_list, axis=0) if x_test_list else np.empty((0,))
        y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,))
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,))
        
        return x_train, x_test, y_train, y_test
    else:
        # Fallback to the original split behavior if needed
        x_train_list, x_test_list = [], []
        y_train_list, y_test_list = [], []
        
        print("\nSplitting dataset (80% train, 20% test sequentially)...")
        for idx in sorted(X_by_label.keys()):
            X = X_by_label[idx]
            n_samples = len(X)
            split_point = int(n_samples * 0.5)
            
            x_train_list.append(X[:split_point])
            x_test_list.append(X[split_point:])
            y_train_list.append(np.full(split_point, idx, dtype=np.int64))
            y_test_list.append(np.full(n_samples - split_point, idx, dtype=np.int64))
            print(f"  Class '{INV_LABEL_MAP.get(idx, str(idx))}': Train={split_point}, Test={n_samples - split_point}")
            
        x_train = np.concatenate(x_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        return x_train, x_test, y_train, y_test


def preprocess_csi_fusion(X_complex, eps=2.0):
    N = len(X_complex)
    X_amp = np.abs(X_complex)
    X_phase = np.angle(X_complex)
    
    # Vectorized Linear Phase Calibration to remove CFO and SFO phase noise
    unwrapped = np.unwrap(X_phase, axis=2)
    x = np.arange(114)
    x_mean = x.mean()
    x_dev = x - x_mean
    D = np.sum(x_dev**2)
    
    Y_mean = unwrapped.mean(axis=2, keepdims=True)
    y_dev = unwrapped - Y_mean
    a = np.sum(y_dev * x_dev, axis=2, keepdims=True) / D
    X_phase_cal = y_dev - a * x_dev
    
    # Apply SR-Std to both amplitude and calibrated phase
    X_amp_norm = np.zeros_like(X_amp)
    X_phase_norm = np.zeros_like(X_phase_cal)
    
    for i in range(N):
        mean_amp = X_amp[i].mean(axis=0, keepdims=True)
        std_amp = X_amp[i].std(axis=0, keepdims=True)
        X_amp_norm[i] = (X_amp[i] - mean_amp) / (std_amp + eps)
        
        mean_phase = X_phase_cal[i].mean(axis=0, keepdims=True)
        std_phase = X_phase_cal[i].std(axis=0, keepdims=True)
        X_phase_norm[i] = (X_phase_cal[i] - mean_phase) / (std_phase + eps)
        
    # Concatenate amplitude and phase features along the subcarrier axis (50, 114) + (50, 114) -> (50, 228)
    X_combined = np.concatenate([X_amp_norm, X_phase_norm], axis=2)
    return X_combined

def preprocess_csi_amp_only(X_complex, eps=3.0):
    N = len(X_complex)
    X_amp = np.abs(X_complex)
    X_amp_norm = np.zeros_like(X_amp)
    
    for i in range(N):
        mean_amp = X_amp[i].mean(axis=0, keepdims=True)
        std_amp = X_amp[i].std(axis=0, keepdims=True)
        X_amp_norm[i] = (X_amp[i] - mean_amp) / (std_amp + eps)
        
    return X_amp_norm
