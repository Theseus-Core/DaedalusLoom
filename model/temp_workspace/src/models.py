import torch
import torch.nn as nn

class Advanced1DCNN(nn.Module):
    def __init__(self, n_subcarriers=228, num_classes=4):
        super(Advanced1DCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.AdaptiveAvgPool1d(4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # input shape: (batch, channels, time)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=5700, num_classes=4): # 50 * 114 = 5700, 50 * 228 = 11400
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # input shape: (batch, channels, time)
        return self.net(x)

class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_dim=228, hidden_dim=128, num_layers=2, num_classes=4):
        super(LSTMGestureClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # input shape: (batch, channels, time)
        # nn.LSTM expects: (batch, time, channels)
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        # Concatenate bidirectional hidden states of the last layer
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(last_hidden)

class SpectrogramCNN2D(nn.Module):
    def __init__(self, in_freq=114, in_time_feat=50, num_classes=4, dropout=0.5):
        super(SpectrogramCNN2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # input shape: (batch, channels, time) = (batch, 114, 50)
        # unsqueeze to (batch, 1, 114, 50)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1DGesture(nn.Module):
    def __init__(self, n_subcarriers=228, num_classes=4):
        super(ResNet1DGesture, self).__init__()
        self.in_channels = 64
        self.prep = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer1 = ResBlock1D(64, 64, stride=1)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)
        return self.classifier(x)

class CNN_GRU_Classifier(nn.Module):
    def __init__(self, n_subcarriers=228, hidden_dim=128, num_classes=4):
        super(CNN_GRU_Classifier, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_subcarriers, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16)
        )
        
        self.gru = nn.GRU(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.permute(0, 2, 1)
        out_gru, _ = self.gru(features)
        out_last = out_gru[:, -1, :]
        return self.classifier(out_last)

