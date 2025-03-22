from sentence_transformers import SentenceTransformer
from src.utils import KNOWLEDGE_BASE
import sqlite3
import numpy as np
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def cosine_similarity(v1, v2):
    """计算两个向量之间的余弦相似度"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


def search(query, top_k=1):
    # 连接到数据库
    conn = sqlite3.connect(f'{KNOWLEDGE_BASE}/knowledge_base.db')
    cursor = conn.cursor()

    # 将查询转换为向量
    query_vector = model.encode(query)

    # 获取所有存储的embeddings
    cursor.execute('SELECT filename, embedding FROM embeddings')
    results = cursor.fetchall()

    # 计算相似度并排序
    similarities = []
    for filename, embedding_bytes in results:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        similarity = cosine_similarity(query_vector, embedding)  # 使用余弦相似度
        similarities.append((filename, similarity))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 关闭连接
    conn.close()

    # 返回前 top_k 个结果
    return similarities[:top_k]


if __name__ == "__main__":
    # 测试搜索功能
    query = "空气相对湿度"
    results = search(query)
    print(f"Top {len(results)} results for query: '{query}'")
    for filename, similarity in results:
        print(f"File: {filename}, Similarity: {similarity:.4f}")
