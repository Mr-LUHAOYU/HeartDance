def calculate_relative_humidity(theta=None, tau=None, p=None, p_theta=None, p_tau=None):
    """
    计算空气相对湿度

    参数：
    theta : 穯干球温度 (单位：℃)，默认值为 None
    tau : 空气湿球温度 (单位：℃)，默认值为 None
    p : 大气压力 (单位：kPa)，默认值为 None
    p_theta : 空气温度等于θ℃时的饱和水蒸气分压力 (单位：kPa)，默认值为 None
    p_tau : 空气温度等于τ℃时的饱和水蒸气分压力 (单位：kPa)，默认值为 None

    返回：空气相对湿度 φ，如果任何输入为 None，则返回 None
    """
    if theta is None:
        return ValueError("空气干球温度,不可以为空值。")
    elif tau is None:
        return ValueError("空气湿球温度,不可以为空值。")
    elif p is None:
        return ValueError("大气压力,不可以为空值。")
    elif p_theta is None:
        return ValueError("空气温度等于θ℃时的饱和水蒸气分压力,不可以为空值。")
    elif p_tau is None:
        return ValueError("空气温度等于τ℃时的饱和水蒸气分压力,不可以为空值。")
    
    # 计算空气相对湿度
    phi = (p_theta - 0.000662 * p * (theta - tau)) / p_tau
    return phi