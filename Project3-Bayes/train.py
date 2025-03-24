import os
import matplotlib.pyplot as plt
import pandas as pd
import bnlearn as bn
import time
import numpy as np
import tqdm


class BayesianAnomalyDetector:
    def __init__(self, model_path="./bnlearn_model.pkl"):
        self.model_path = model_path
        self.model = None

    def train_model(self, train_data_path="data.csv"):
        """模型训练：结构学习+参数学习"""
        df = pd.read_csv(train_data_path)

        # 结构学习
        DAG = bn.structure_learning.fit(df, methodtype='hc', n_jobs=8)
        print('------------结构学习完成----------------')
        # 参数学习
        self.model = bn.parameter_learning.fit(DAG, df, methodtype='ml', n_jobs=8)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        bn.save(self.model, filepath=self.model_path)
        bn.plot(self.model)
        plt.show()
        print("[INFO] 模型训练完成并保存")

    def infer_single(self, evidence: dict, ):
        # input('这里是Infer...')
        # print(evidence)
        result = []
        ground_truth = []
        for node, truth in evidence.items():
            cur_evidence = evidence.copy()
            del cur_evidence[node]
            try:
                res = bn.inference.fit(
                    self.model,
                    variables=[node],
                    evidence=cur_evidence,
                    verbose=0
                )
                result.append(self._parse_result(res, node))
            except KeyError:
                result.append(1)

            ground_truth.append(truth)
        # input('请输入...')
        result = np.array(result)
        ground_truth = np.array(ground_truth)
        return np.sqrt(np.sum((result - ground_truth) ** 2))

    def infer(self, test_data):
        """异常推理"""
        if not self.model:
            self.model = bn.load(self.model_path)
        print(self.model['model'].states)

        results = {}
        for case_id, evidence in tqdm.tqdm(test_data.iterrows(), total=len(test_data), desc='Processing items'):
            # 过滤无效证据
            valid_evidence = {k: v for k, v in evidence.items() if k in self.model['model'].states}
            if not valid_evidence: continue

            # 执行推理
            result = self.infer_single(valid_evidence)
            results[case_id] = result

        return results

    def _parse_result(self, result, variables):
        """解析推理结果"""
        states = result.state_names[variables]
        return states[0]

    def evaluate(self, results, threshold=0.8):
        """评估模型性能"""
        tp, fp, total = 0, 0, 0
        start_time = time.time()
        MIN_VAL = float('inf')
        MAX_VAL = float('-inf')

        for case_id, loss in results.items():
            total += 1
            MAX_VAL = max(MAX_VAL, loss)
            MIN_VAL = min(MIN_VAL, loss)
            if loss >= threshold:
                tp += 1
            else:
                fp += 1

        time_cost = time.time() - start_time
        print(f"- 正常判定准确率: {tp / (tp + fp):.2%}" if (tp + fp) else "无异常判定")
        print(f"- 平均推理时间: {time_cost / total:.2f}s" if total else "无有效数据")
        print(MAX_VAL, MIN_VAL)
        return {'tp': tp, 'fp': fp, 'time': time_cost}


if __name__ == "__main__":
    # 实例化检测器
    detector = BayesianAnomalyDetector()

    # 模型训练
    # detector.train_model()

    # 加载测试数据（假设read_test返回{case_id: {evidence_dict}}格式）
    test_data = pd.read_csv('metrics_abnormal.csv')
    # test_data['normal'] = 1

    origin_data = pd.read_csv('data_origin.csv')
    for col in origin_data.columns:
        if origin_data[col].nunique() == 1:
            test_data.drop(col, axis=1, inplace=True)
        else:
            test_data[col] = (test_data[col]-origin_data[col].min())/(origin_data[col].max()-origin_data[col].min())

    # 执行推理
    results = detector.infer(test_data)

    # 评估结果
    detector.evaluate(results, threshold=0.95)
