import bnlearn as bn
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

model_path = '../Project3-AnomalyDetection/bnlearn_model.pkl'  # 模型路径
file_path = "../Project3-AnomalyDetection/metrics_abnormal.csv"  # 测试数据路径

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

evidence_list = list(parent_nodes)  # 存储父节点（证据）
target_list = list(leaf_nodes)  # 存储子节点（目标）

test_data = pd.read_csv(file_path)[:1000]

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(test_data)
normalized_df = pd.DataFrame(normalized_data, columns=test_data.columns)
normalized_df_scaled = normalized_df.round(15)


def inference_fit(model, evidence, variables, target_values):
    """
    贝叶斯推理-基于贝叶斯网络
    :param model: 贝叶斯网络模型
    :param evidence: 父节点数据
    :param variables: 预测节点
    :return:{节点A: [1, 0.20353982], 节点B: [0, 0.08849558]} 1/0:根节点预测与监控数据是否一致 0.20353982: 概率值
    """
    start_time = time.time()  # 记录开始时间
    score = {}
    for variable in variables:
        # 贝叶斯推理-基于贝叶斯网络
        result = bn.inference.fit(model, evidence=evidence, variables=[variable], verbose=0)
        if variable not in result.state_names:
            continue
        if target_values[variable] not in result.state_names[variable]:
            continue
        # 使用 numpy 的 argmax() 函数获取最大概率值的索引
        max_index = result.values.argmax()
        # 获取最大概率对应的状态名称
        max_state = result.state_names[variable][max_index]

        if target_values[variable] == max_state:
            score[variable] = [1, result.values[max_index]]
        else:
            score[variable] = [0, result.values[max_index]]

    end_time = time.time()  # 记录结束时间
    # 计算每次推理的耗时
    inference_time = end_time - start_time

    return score, inference_time


def retrospect(score: dict, _evidence: dict = None, result: dict = None, edges_dict: dict = None,
               inference_fit_func=None):
    """
    追溯异常父节点
    :param result: 结果
    :param _evidence: 父节点数据
    :param score: 预测结果根节点列表 {节点A: [1, 0.20353982], 节点B: [0, 0.08849558]}
    :param edges_dict: 边信息，包含节点与父节点的关系
    :param inference_fit_func: 推理函数（根据实际需要传入）
    :return: {节点: 概率值}
    """
    if not _evidence:
        _evidence = {}  # 如果没有传入 _evidence，设置为空字典
    if not result:
        result = {}
    target_values_father = 0
    father = 0
    # 校验score
    if len([True for value in score.values() if value[0] == 0]) == 0:
        return result  # 如果没有错误预测，直接返回结果
    else:
        # 输出错误的预测
        print("预测错误的节点:")
        for key, value in score.items():
            if value[0] == 0:  # 0 表示预测错误
                print(f"节点: {key}, 概率值: {value[1]}")
                # 输出该节点的父节点
                if key in edges_dict:
                    parent_nodes = edges_dict[key]
                    if parent_nodes:
                        father = parent_nodes[0]
                        print(f"导致预测错误的父节点: {parent_nodes}")
                    else:
                        break
                    for parent in parent_nodes:
                        if parent in _evidence:
                            target_values_father = _evidence[parent]

    target_values_father = {father: target_values_father}

    _variables = []
    for key, value in score.items():
        if value[0] == 0 and key in edges_dict.keys():
            # 如果预测错误，并且有父节点
            _variables += [i for i in edges_dict[key] if i in _evidence.keys()]

    _variables = list(set(_variables))  # 去重
    variables = {}
    evidence = {}

    # 将所有父节点和其他信息分类
    for i in list(set(_variables)):
        variables[i] = _evidence[i]

    for key, value in _evidence.items():
        if key in _variables:
            variables[key] = value
        else:
            evidence[key] = value


total_correct_node = 0
total_node = 0


def is_correct(score):
    global total_correct_node  # 使用全局变量
    global total_node  # 使用全局变量
    current_correct_node = 0
    current_node = 0
    for _, results in score.items():
        total_node += 1
        current_node += 1
        if results[0] == 1:
            total_correct_node += 1
            current_correct_node += 1
        if results[0] != 1:  # 如果任何一个目标预测不一致，标记为错误
            continue  # 只要一个目标预测不一致，就跳出当前样本的判断
    accuracy = current_correct_node / current_node if current_node > 0 else 0
    # print(accuracy)
    if accuracy >= 0.8:  # 如果所有目标都一致，认为该样本预测正确
        return True


total_accuracies = 0
total_error = 0
total_samples = len(normalized_df_scaled)
result = {}

edges_dict = {}
for node in model.nodes():
    parents = list(model.get_parents(node))  # 获取每个节点的父节点
    edges_dict[node] = parents

total_inference_time = 0  # 累加所有推理耗时
count = 0
# 在主程序中调用这个推理函数
for index, row in normalized_df_scaled.iterrows():
    # 从 evidence_list 和 target_list 获取对应的证据和目标
    evidence = {evi: row[evi] for evi in evidence_list if row[evi] in valid_states[evi]}
    target_values = {tar: row[tar] for tar in target_list}
    # 使用推理函数来进行推理  推理时间
    inference_result, inference_time = inference_fit(model_dict, evidence, target_list, target_values)

    # 总的推理时间
    total_inference_time += inference_time

    if is_correct(inference_result):
        total_accuracies += 1
    else:
        total_error += 1

    count += 1
    print('\r', count, end='')
print()
accuracy = total_accuracies / total_samples if total_samples > 0 else 0
accuracy_node = total_correct_node / total_node if total_samples > 0 else 0
mean_time = total_inference_time / total_samples if total_samples > 0 else 0

print("节点预测正确总数:", total_correct_node, "节点总数:", total_node, "节点准确率:", accuracy_node)
print("预测正确总数:", total_accuracies, "样本总数:", total_samples, "准确率:", accuracy)
print("总推理时间:", total_inference_time, "平均推理时间:", mean_time)
