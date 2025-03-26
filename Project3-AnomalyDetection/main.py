import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_train = pd.read_csv('data_origin.csv')

data_test1 = pd.read_csv('metrics_abnormal.csv')
y_label1 = np.array([1] * len(data_test1))
data_test2 = pd.read_csv('metrics_anomaly.csv')
y_label2 = np.array([0] * len(data_test2))

scaler = MinMaxScaler()
data_train = scaler.fit_transform(data_train)
data_test1 = scaler.fit_transform(data_test1)
data_test2 = scaler.fit_transform(data_test2)

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


args = {
    "encoding_dim": 36,
    "num_epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-3
}


class AnomalyDetection(object):
    def __init__(self, encoding_dim, learning_rate=1e-3, **kwargs):
        input_dim = X_train.shape[1]
        self.model = Autoencoder(input_dim, encoding_dim).to(device=device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, X, batch_size, num_epochs, **kwargs):
        # 训练过程
        for epoch in range(num_epochs):
            self.model.train()
            for i in range(0, len(X), batch_size):
                batch = X[i:i + batch_size]

                output = self.model(batch)
                loss = self.criterion(output, batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.eval_error(X)

    def eval_error(self, X):
        self.model.eval()
        with torch.no_grad():
            X_recon = self.model(X)
            error = torch.abs(X - X_recon).squeeze()
            self.std = torch.std(error, dim=0)
            self.mean = torch.mean(error, dim=0)
            # print(self.std)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_pred = self.model(X)

        loss = [self.criterion(output, batch).item() for output, batch in zip(X_pred, X)]

        threshold = np.percentile(loss, 50)
        anomalies = np.array(loss) > threshold

        return anomalies

    def locate_anomalous_features(self, X):
        self.model.eval()
        with torch.no_grad():
            X_recon = self.model(X)
            error = torch.abs(X - X_recon)
        # ranked_features = torch.argsort(distinctiveness, descending=True)
        ranked_features, _ = torch.sort(error, dim=1, descending=True)
        print(ranked_features)
        return ranked_features.tolist()


def get_anomalous_features():
    model = AnomalyDetection(**args)
    model.train(X_train, **args)
    pd.DataFrame(model.locate_anomalous_features(X_test)).to_csv('test.csv')


def search_best_param():
    result = pd.DataFrame(columns=['encoding-dim', 'num-epochs', 'batch-size', 'lr', 'Accuracy'])
    done = 0
    for encoding_dim in range(10, 41):
        for batch_size in [16, 32, 64, 128]:
            for lr in [1e-3, 1e-4, 1e-5]:
                model = AnomalyDetection(encoding_dim=encoding_dim, learning_rate=lr)
                model.train(X_train, batch_size=batch_size, num_epochs=30)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_label, y_pred)
                info = pd.DataFrame({
                    'encoding-dim': encoding_dim,
                    'num-epochs': 30,
                    'batch-size': batch_size,
                    'lr': lr,
                    'Accuracy': acc
                })
                result = pd.concat([result, info], ignore_index=True)
                print(f'Encoding dim: {encoding_dim}, Batch size: {batch_size}, Learning rate: {lr}, done: {done}.')

    result.to_csv('result3.csv')
