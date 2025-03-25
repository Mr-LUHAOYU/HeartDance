import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_train = pd.read_csv('data_origin.csv')

data_test1 = pd.read_csv('metrics_abnormal.csv')
y_label1 = np.array([1] * len(data_test1))
data_test2 = pd.read_csv('metrics_anomaly.csv')
y_label2 = np.array([0] * len(data_test2))

# 标准化数据
scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test1 = scaler.fit_transform(data_test1)
data_test2 = scaler.fit_transform(data_test2)

data_test = pd.concat([pd.DataFrame(data_test1), pd.DataFrame(data_test2)]).to_numpy()
y_label = np.append([], [y_label1, y_label2])

# 将数据转换为 PyTorch 张量
X_train = torch.tensor(data_train, dtype=torch.float32, device=device)
X_test = torch.tensor(data_test, dtype=torch.float32, device=device)
y_label = torch.tensor(y_label, dtype=torch.float32, device=device)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, encoding_dim),
            nn.LayerNorm(encoding_dim),
            nn.ReLU()
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.LayerNorm(encoding_dim),
            nn.Linear(encoding_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.Sigmoid()  # 使用 Sigmoid 将输出限制在 [0, 1] 范围内
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train8test(encoding_dim=20, num_epochs=30, batch_size=16, lr=0.001):
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim)
    model.to(device=device)

    criterion = nn.MSELoss()
    L1_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练过程
    for epoch in trange(num_epochs):
        model.train()
        LOSS = 0
        for i in range(0, len(X_train), batch_size):
            # 获取小批量数据
            batch = X_train[i:i + batch_size]
            # 前向传播
            output = model(batch)
            loss = criterion(output, batch) + L1_criterion(output, batch)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS += loss.item()

    model.eval()

    with torch.no_grad():
        X_test_pred = model(X_test)
    loss_test = [criterion(output, batch).item() for output, batch in zip(X_test_pred, X_test)]

    threshold = np.percentile(loss_test, 50)
    anomalies = np.array(loss_test) > threshold
    tn, fp, fn, tp = confusion_matrix(y_label.cpu().numpy(), anomalies).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return pd.DataFrame({
        'encoding-dim': [encoding_dim],
        'Accuracy': [accuracy],
        'Recall': [recall],
        'Precision': [precision],
        'F1-score': [f1],
    })


result = pd.DataFrame(columns=['encoding-dim', 'Accuracy', 'Recall', 'Precision', 'F1-score'])
for encoding_dim in range(16, 61):
    info = train8test(encoding_dim=encoding_dim)
    result = pd.concat([result, info], ignore_index=True)

result.to_csv('result1.csv')
result.plot()
plt.show()
