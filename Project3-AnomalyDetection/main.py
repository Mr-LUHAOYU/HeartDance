import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_train = pd.read_csv('data_origin.csv')

data_test1 = pd.read_csv('metrics_abnormal.csv')
y_label1 = np.array([1] * len(data_test1))
data_test2 = pd.read_csv('metrics_anomaly.csv')
y_label2 = np.array([0] * len(data_test2))

# scaler = MinMaxScaler()
# data_train = scaler.fit_transform(data_train)
# data_test1 = scaler.fit_transform(data_test1)
# data_test2 = scaler.fit_transform(data_test2)

data_test = pd.concat([pd.DataFrame(data_test1), pd.DataFrame(data_test2)]).to_numpy()
y_label = np.append([], [y_label1, y_label2])

X_train = torch.tensor(data_train, dtype=torch.float32).to(device=device)
X_test = torch.tensor(data_test, dtype=torch.float32).to(device=device)
y_label = torch.tensor(y_label, dtype=torch.float32).to(device=device)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.3):
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
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train8test(
        encoding_dim=20, num_epochs=30,
        batch_size=16, lr=0.001,
):
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim).to(device=device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    result_list = []

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i + batch_size]

            output = model(batch)
            loss = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 9:
            model.eval()

            with torch.no_grad():
                X_test_pred = model(X_test)

            loss_test = [criterion(output, batch).item() for output, batch in zip(X_test_pred, X_test)]

            threshold = np.percentile(loss_test, 50)
            anomalies = np.array(loss_test) > threshold
            accuracy = accuracy_score(y_label.cpu().numpy(), anomalies)

            result_list.append([encoding_dim, epoch + 1, batch_size, lr, accuracy])

    return pd.DataFrame(result_list, columns=['encoding-dim', 'num-epochs', 'batch-size', 'lr', 'Accuracy'])


# result = pd.DataFrame(columns=['encoding-dim', 'num-epochs', 'batch-size', 'lr', 'Accuracy'])
# done = 0
# for encoding_dim in range(10, 41):
#     for batch_size in [16, 32, 64, 128]:
#         for lr in [1e-3, 1e-4, 1e-5]:
#             info = train8test(encoding_dim=encoding_dim, batch_size=batch_size, lr=lr)
#             result = pd.concat([result, info], ignore_index=True)
#             print(f'Encoding dim: {encoding_dim}, Batch size: {batch_size}, Learning rate: {lr}, done: {done}.')
#
# result.to_csv('result3.csv')

print(train8test(encoding_dim=36, batch_size=16, num_epochs=30, lr=1e-3))
