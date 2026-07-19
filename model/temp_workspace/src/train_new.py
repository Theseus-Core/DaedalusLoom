# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Adjust path to import dataset and models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import dataset
import models
import config

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----------------- Transformer Architecture -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CSITransformer(nn.Module):
    def __init__(self, input_dim=228, d_model=128, nhead=4, num_layers=2, num_classes=len(config.LABEL_MAP)):
        super(CSITransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # input shape: (batch, channels, time) = (batch, 228, 50)
        x = x.permute(0, 2, 1) # -> (batch, 50, 228)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling over temporal axis
        return self.classifier(x)

# ----------------- Training Pipeline -----------------
def train_configuration(config_name, model, train_loader, test_loader, epochs, device, num_classes=None):
    if num_classes is None:
        num_classes = len(config.LABEL_MAP)
    print(f"\n================ Training Configuration: {config_name} ================")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_test_acc = 0.0
    best_weights = None
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        model.train()
        correct, total, loss_val = 0, 0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_val += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_correct, test_total = 0, 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                for c in range(num_classes):
                    mask = (labels == c)
                    class_total[c] += mask.sum().item()
                    class_correct[c] += ((predicted == labels) & mask).sum().item()
                
        test_acc = 100 * test_correct / test_total
        avg_loss = loss_val / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_weights = model.state_dict().copy()
            
        class_accs = []
        for c in range(num_classes):
            if class_total[c] > 0:
                acc = 100 * class_correct[c] / class_total[c]
                class_accs.append(f"Class {c}={acc:.2f}%")
            else:
                class_accs.append(f"Class {c}=N/A")
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}% | {' | '.join(class_accs)}")
        
    # Evaluate best model weights on test set to get final per-class accuracies
    model.load_state_dict(best_weights)
    model.eval()
    best_class_correct = [0] * num_classes
    best_class_total = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            for c in range(num_classes):
                mask = (labels == c)
                best_class_total[c] += mask.sum().item()
                best_class_correct[c] += ((predicted == labels) & mask).sum().item()
                
    print(f"Finished {config_name}! Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"\n--- Final Best Per-Class Accuracy Summary for {config_name} ---")
    best_class_accuracies = {}
    for c in range(num_classes):
        if best_class_total[c] > 0:
            acc = 100 * best_class_correct[c] / best_class_total[c]
            best_class_accuracies[c] = acc
            print(f"  Class {c}: {best_class_correct[c]}/{best_class_total[c]} = {acc:.2f}%")
        else:
            best_class_accuracies[c] = 0.0
            print(f"  Class {c}: 0/0 = N/A")
            
    return best_test_acc, best_weights, history, best_class_accuracies

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Establish organized directories
    src_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(src_dir)
    models_dir = os.path.join(workspace_dir, "models")
    plots_dir = os.path.join(workspace_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load raw dataset using dataset.py
    X_by_label = dataset.load_dataset()
    x_train_raw, x_test_raw, y_train, y_test = dataset.split_80_20(X_by_label)
    
    # 2. Preprocess: Fused Amplitude + Calibrated Phase
    print("\n--- Preprocessing: Amplitude + Calibrated Phase Fusion ---")
    x_train_fusion = dataset.preprocess_csi_fusion(x_train_raw)
    x_test_fusion = dataset.preprocess_csi_fusion(x_test_raw)
    
    # Permute to (N, channels, time) = (N, 228, 50)
    x_train_t = torch.tensor(x_train_fusion, dtype=torch.float32).permute(0, 2, 1)
    x_test_t = torch.tensor(x_test_fusion, dtype=torch.float32).permute(0, 2, 1)
    
    train_loader = DataLoader(TensorDataset(x_train_t, torch.tensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test_t, torch.tensor(y_test)), batch_size=64, shuffle=False)
    
    epochs = 30
    all_results = {}
    num_classes = len(config.LABEL_MAP)
    
    # =========================================================================
    # Config 1: Simple MLP (Fused Amp + Phase)
    # =========================================================================
    model_mlp = models.SimpleMLP(input_dim=11400, num_classes=num_classes).to(device)
    best_acc_mlp, weights_mlp, history_mlp, class_accs_mlp = train_configuration(
        "MLP", model_mlp, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["MLP"] = (best_acc_mlp, class_accs_mlp)
    torch.save(weights_mlp, os.path.join(models_dir, "best_new_mlp.pth"))
    print(f"Saved MLP model to {os.path.join(models_dir, 'best_new_mlp.pth')}")
    
    # =========================================================================
    # Config 2: CNN1D (Fused Amp + Phase)
    # =========================================================================
    model_cnn = models.Advanced1DCNN(n_subcarriers=228, num_classes=num_classes).to(device)
    best_acc_cnn, weights_cnn, history_cnn, class_accs_cnn = train_configuration(
        "CNN1D", model_cnn, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["CNN1D"] = (best_acc_cnn, class_accs_cnn)
    torch.save(weights_cnn, os.path.join(models_dir, "best_optimized_cnn.pth"))
    print(f"Saved CNN1D model to {os.path.join(models_dir, 'best_optimized_cnn.pth')}")
    
    # =========================================================================
    # Config 3: ResNet1D (Fused Amp + Phase)
    # =========================================================================
    model_resnet = models.ResNet1DGesture(n_subcarriers=228, num_classes=num_classes).to(device)
    best_acc_resnet, weights_resnet, history_resnet, class_accs_resnet = train_configuration(
        "ResNet1D", model_resnet, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["ResNet1D"] = (best_acc_resnet, class_accs_resnet)
    torch.save(weights_resnet, os.path.join(models_dir, "best_new_resnet.pth"))
    print(f"Saved ResNet1D model to {os.path.join(models_dir, 'best_new_resnet.pth')}")
    
    # =========================================================================
    # Config 4: CNN1D + GRU (Fused Amp + Phase)
    # =========================================================================
    model_cnngru = models.CNN_GRU_Classifier(n_subcarriers=228, num_classes=num_classes).to(device)
    best_acc_cnngru, weights_cnngru, history_cnngru, class_accs_cnngru = train_configuration(
        "CNN1D-GRU", model_cnngru, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["CNN1D-GRU"] = (best_acc_cnngru, class_accs_cnngru)
    torch.save(weights_cnngru, os.path.join(models_dir, "best_new_cnngru.pth"))
    print(f"Saved CNN1D-GRU model to {os.path.join(models_dir, 'best_new_cnngru.pth')}")
    
    # =========================================================================
    # Config 5: CSITransformer (Fused Amp + Phase)
    # =========================================================================
    model_trans = CSITransformer(input_dim=228, num_classes=num_classes).to(device)
    best_acc_trans, weights_trans, history_trans, class_accs_trans = train_configuration(
        "CSITransformer", model_trans, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["CSITransformer"] = (best_acc_trans, class_accs_trans)
    torch.save(weights_trans, os.path.join(models_dir, "best_new_transformer.pth"))
    print(f"Saved CSITransformer model to {os.path.join(models_dir, 'best_new_transformer.pth')}")

    # =========================================================================
    # Config 6: CNN2D (Fused Amp + Phase)
    # =========================================================================
    model_cnn2d = models.SpectrogramCNN2D(in_freq=228, in_time_feat=50, num_classes=num_classes).to(device)
    best_acc_cnn2d, weights_cnn2d, history_cnn2d, class_accs_cnn2d = train_configuration(
        "CNN2D", model_cnn2d, train_loader, test_loader, epochs=epochs, device=device, num_classes=num_classes
    )
    all_results["CNN2D"] = (best_acc_cnn2d, class_accs_cnn2d)
    torch.save(weights_cnn2d, os.path.join(models_dir, "best_new_cnn2d.pth"))
    print(f"Saved CNN2D model to {os.path.join(models_dir, 'best_new_cnn2d.pth')}")

    # =========================================================================
    # Final Summary Table Printing
    # =========================================================================
    print("\n" + "=" * 110)
    print("                                  WIFI CSI MODEL COMPARISON SUMMARY")
    print("=" * 110)
    header = f"{'Model Name':<16} | {'Overall Acc':<12} | " + " | ".join([f"Class {c} ({config.INV_LABEL_MAP_EN[c]})" for c in range(num_classes)])
    print(header)
    print("-" * 110)
    for model_name, (overall_acc, class_accs) in all_results.items():
        class_str = " | ".join([f"{class_accs[c]:6.2f}%" for c in range(num_classes)])
        print(f"{model_name:<16} | {overall_acc:>10.2f}% | {class_str}")
    print("=" * 110)
 
if __name__ == '__main__':
    main()