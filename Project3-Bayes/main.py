import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 加载数据
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
X_train = torch.tensor(data_train, dtype=torch.float32)
X_test = torch.tensor(data_test, dtype=torch.float32)


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


# 初始化模型
input_dim = X_train.shape[1]
encoding_dim = 20  # 编码层的维度，可以根据需要调整
model = Autoencoder(input_dim, encoding_dim)

criterion = nn.MSELoss()  # 使用均方误差作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器

# 训练参数
num_epochs = 30
batch_size = 16

# 训练过程
for epoch in range(num_epochs):
    model.train()
    LOSS = 0
    for i in range(0, len(X_train), batch_size):
        # 获取小批量数据
        batch = X_train[i:i + batch_size]
        # 前向传播
        output = model(batch)
        loss = criterion(output, batch)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        LOSS += loss.item()
    # 打印训练损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {LOSS:.4f}')

# 切换到评估模式
model.eval()

# 获取重构误差阈值
with torch.no_grad():
    X_train_pred = model(X_train)
mse = [criterion(output, batch).item() for output, batch in zip(X_train_pred, X_train)]
mse = np.array(mse)
print(mse.min(), mse.max(), np.percentile(mse, 95))

# 对测试集进行预测
with torch.no_grad():
    X_test_pred = model(X_test)

# 计算重构误差（均方误差）
# mse = torch.mean((X_test - X_test_pred) ** 2, dim=1).numpy()
mse = [criterion(output, batch).item() for output, batch in zip(X_test_pred, X_test)]
mse = np.array(mse)
# mse = criterion(X_test_pred, X_test)
# mse = np.sqrt(mse)
print(mse)
threshold = np.percentile(mse, 50)
print(threshold)

# 检测异常
anomalies = mse > threshold

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_label, anomalies
)
print(cm)

# 可视化重构误差
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50)
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2)
plt.xlabel("Reconstruction error")
plt.ylabel("Frequency")
plt.title("Reconstruction error distribution")
plt.show()

# 保存模型
torch.save(model.state_dict(), 'autoencoder_model.pth')
