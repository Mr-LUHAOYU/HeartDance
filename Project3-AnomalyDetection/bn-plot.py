import bnlearn as bn
import pickle
from pyvis.network import Network

# 1. 从pkl文件加载模型
with open('bnlearn_model.pkl', 'rb') as f:
    model = pickle.load(f)


# 2. 创建交互式网络可视化
def save_interactive_bn(model, filename='bn_network.html'):
    # 使用pyvis创建网络
    net = Network(height='800px', width='100%', directed=True, notebook=False)

    # 添加节点和边
    for node in model['model'].nodes():
        net.add_node(node, label=node, shape='circle', size=160)

    for edge in model['model'].edges():
        net.add_edge(edge[0], edge[1], arrows='to')

    # 设置物理布局使图形更美观
    net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200)

    # 保存为HTML文件
    net.show(filename)
    print(f"交互式网络已保存为: {filename}")


# 3. 生成并保存交互式可视化
save_interactive_bn(model, 'interactive_bn.html')

