import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
            nn.Softmax()
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
L1_criterion = nn.L1Loss()
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
        loss = criterion(output, batch) + L1_criterion(output, batch)
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
loss_train = [criterion(output, batch).item() for output, batch in zip(X_train_pred, X_train)]
loss_train = np.array(loss_train)
# print(mse.min(), mse.max(), np.percentile(mse, 95))

# 对测试集进行预测
with torch.no_grad():
    X_test_pred = model(X_test)

# 计算重构误差（均方误差）
loss_test = [criterion(output, batch).item() for output, batch in zip(X_test_pred, X_test)]
loss_test = np.array(loss_test)


# 检测异常
def show(threshold):
    print(threshold)
    anomalies = loss_test > threshold

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        y_label, anomalies
    )
    print(cm)


show(np.percentile(loss_train, 95))
show(np.percentile(loss_test, 50))

# 保存模型
torch.save(model.state_dict(), 'autoencoder_model.pth')
