import bnlearn as bn
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model_path = '../Project3-AnomalyDetection/bnlearn_model.pkl'  # 模型路径
file_path = "../Project3-AnomalyDetection/metrics_anomaly.csv"  # 测试数据路径

model_dict = bn.load(filepath=model_path)
model = model_dict['model']
valid_states = model.states

all_nodes = set(model.nodes())
parent_nodes = set()
for edge in model.edges():
    parent_nodes.add(edge[0])

leaf_nodes = all_nodes - parent_nodes

print("父节点（有子节点的节点）:", parent_nodes)
print("子节点（没有子节点的节点）:", leaf_nodes)

evidence_list = list(parent_nodes)
target_list = list(leaf_nodes)

test_data = pd.read_csv(file_path)

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(test_data)
normalized_df = pd.DataFrame(normalized_data, columns=test_data.columns)


def inference_fit(model, evidence, variables, target_values):
    start_time = time.time()
    score = {}
    for variable in variables:
        result = bn.inference.fit(model, evidence=evidence, variables=[variable], verbose=0)
        if variable not in result.state_names:
            continue
        if target_values[variable] not in result.state_names[variable]:
            continue

        max_index = result.values.argmax()
        max_state = result.state_names[variable][max_index]

        if target_values[variable] == max_state:
            score[variable] = [1, result.values[max_index]]
        else:
            score[variable] = [0, result.values[max_index]]

    end_time = time.time()
    inference_time = end_time - start_time

    return score, inference_time


total_correct_node = 0
total_node = 0


def is_correct(score):
    global total_correct_node, total_node
    current_correct_node = 0
    current_node = 0
    for _, results in score.items():
        total_node += 1
        current_node += 1
        if results[0] == 1:
            total_correct_node += 1
            current_correct_node += 1
        if results[0] != 1:
            continue
    accuracy = current_correct_node / current_node
    return accuracy >= 0.8


total_accuracies = 0
total_error = 0
total_samples = len(normalized_df)
result = {}

edges_dict = {}
for node in model.nodes():
    parents = list(model.get_parents(node))  # 获取每个节点的父节点
    edges_dict[node] = parents

total_inference_time = 0  # 累加所有推理耗时
count = 0

for index, row in normalized_df.iterrows():
    evidence = {evi: row[evi] for evi in evidence_list if row[evi] in valid_states[evi]}
    target_values = {tar: row[tar] for tar in target_list}

    inference_result, inference_time = inference_fit(model_dict, evidence, target_list, target_values)

    total_inference_time += inference_time

    if is_correct(inference_result):
        total_accuracies += 1
    else:
        total_error += 1

    count += 1
    print('\r', count, end='')
print()
accuracy = total_accuracies / total_samples
accuracy_node = total_correct_node / total_node
mean_time = total_inference_time / total_samples

print("节点预测正确总数:", total_correct_node, "节点总数:", total_node, "节点准确率:", accuracy_node)
print("预测正确总数:", total_accuracies, "样本总数:", total_samples, "准确率:", accuracy)
print("总推理时间:", total_inference_time, "平均推理时间:", mean_time)
