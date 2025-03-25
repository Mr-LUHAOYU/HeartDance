import pandas as pd


def process_csv(file_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 遍历每一列
    for col in df.columns:
        # 检查列中的所有值是否相同
        if df[col].nunique() == 1:
            # 如果所有值相同，删除该列
            df.drop(col, axis=1, inplace=True)
        else:
            # 否则进行min-max归一化
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # 保存处理后的CSV文件
    df.to_csv(output_path, index=False)


# 示例调用
process_csv('data_origin.csv', 'data.csv')