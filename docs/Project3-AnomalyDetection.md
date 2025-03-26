# 地铁制动缸泄漏异常检测定位系统


## 1. 实验目标
本实验旨在开发一个基于深度学习的异常检测系统，用于：
1. 根据地铁制动缸100多个传感器的检测值判断是否存在异常
2. 准确定位异常传感器
3. 为维护人员提供可解释的异常特征排名

## 2. 实验方法

### 2.1 数据准备
- 训练数据：`data_origin.csv`（正常工况数据）
- 测试数据：
  - `metrics_abnormal.csv`（异常数据，标签为1）
  - `metrics_anomaly.csv`（正常数据，标签为0）
- 数据预处理：
  - 使用MinMaxScaler进行归一化
  - 合并测试数据并生成对应标签

### 2.2 模型架构
采用自编码器(Autoencoder)模型，结构如下：

**编码器(Encoder)**:

1. LayerNorm层（输入维度）
2. 线性层（input_dim → encoding_dim）
3. LayerNorm层（encoding_dim）
4. ReLU激活函数

**解码器(Decoder)**:
1. LayerNorm层（encoding_dim）
2. 线性层（encoding_dim → input_dim）
3. LayerNorm层（input_dim）
4. Sigmoid激活函数

![AE-3](img-src/AE-3.png)

### 2.3 评价指标

模型采用准确率Accuracy作为评价指标。

经过实验发现，由于测试数据中正常和异常数据的占比相同，精度，召回率等指标都与Accuracy近似。

### 2.4 训练配置
- 设备：自动选择CUDA(GPU)或CPU
- 默认超参数：
  - encoding_dim: 36
  - num_epochs: 30
  - batch_size: 16
  - learning_rate: 1e-3
- 损失函数：均方误差损失(MSELoss)
- 优化器：Adam

### 2.4 异常检测与定位
1. **异常判断**：
   - 计算重构误差
   - 以重构误差的50%分位数作为阈值
   - 超过阈值则判定为异常

2. **异常定位**：
   - 计算各特征的重构误差
   - 按误差大小对特征进行排序
   - 输出异常特征及其得分

## 3. 实验结果

### 3.1 超参数搜索
通过`search_best_param()`函数进行网格搜索，评估不同超参数组合的性能：
- encoding_dim范围：10-40
- batch_size选项：16, 32, 64, 128
- learning_rate选项：1e-3, 1e-4, 1e-5
- epoch选项：10, 20, 30, 40, 50

得到的最佳的性能组合为：

```json
{
    "encoding_dim": 36,
    "num_epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-3
}
```

该组合的性能为90%.

![AE-1](img-src/AE-2.png)详细的指标结果可以参考 [result.csv](https://github.com/Mr-LUHAOYU/HeartDance/blob/main/Project3-AnomalyDetection/result.csv)，下载链接： [result.csv](https://website-lhy.oss-cn-shanghai.aliyuncs.com/result.csv).

### 3.2 异常特征定位

推断出的异常特征案例：

![AE-1](img-src/AE-1.png)

## 4. 使用说明

1. **环境要求**：
   - Python 3.x
   - PyTorch
   - pandas
   - numpy
   - scikit-learn

2. **运行方式**：
   - 调用``get_anomalous_features``函数
   
3. **输入文件**：
   - `data_origin.csv`：训练数据
   - `metrics_abnormal.csv`：异常测试数据
   - `metrics_anomaly.csv`：正常测试数据


## 5. 改进方向

1. **模型优化**：
   - 尝试更复杂的网络结构（如卷积自编码器）
   - 引入注意力机制提高定位精度

2. **数据处理**：
   - 增加数据增强技术
   - 考虑时间序列特性（如使用LSTM自编码器）

3. **评估指标**：
   - 开发专门的异常定位评估指标
   
4. **系统集成**：
   - 开发可视化界面展示异常检测结果
   - 实现实时监测功能

## 6. 结论
本实验成功实现了一个基于自编码器的地铁制动缸泄漏异常检测系统，能够有效识别异常并定位异常传感器。通过超参数搜索可以进一步优化模型性能，为地铁制动系统的预防性维护提供了有力工具。