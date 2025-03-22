def calculate_fracture_stress(E0=None, alpha=None, df=None, d=None, dc=None, dg=None):
    """
    计算每根光纤的断裂应力

    参数：
    E0 : 杨氏模量 (单位：GPa)，默认值为 None
    alpha : 非线性应力-应变特性的修正系数，默认值为 None
    df : 玻璃光纤直径 (单位：微米)，默认值为 None
    d : 光纤断裂时压板间的距离 (单位：微米)，默认值为 None
    dc : 包括任何涂覆层的光纤总直径 (单位：微米)，默认值为 None
    dg : 槽的深度 (单位：微米)，默认值为 None

    返回：光纤的断裂应力 (单位：GPa)，如果任何输入为 None，则返回错误信息
    """
    if E0 is None:
        return ValueError("杨氏模量,不可以为空值。")
    elif alpha is None:
        return ValueError("非线性应力-应变特性的修正系数,不可以为空值。")
    elif df is None:
        return ValueError("玻璃光纤直径,不可以为空值。")
    elif d is None:
        return ValueError("光纤断裂时压板间的距离,不可以为空值。")
    elif dc is None:
        return ValueError("包括任何涂覆层的光纤总直径,不可以为空值。")
    elif dg is None:
        return ValueError("槽的深度,不可以为空值。")

    # 计算 alpha'
    alpha_prime = 0.75 * alpha - 0.25

    # 计算 εf
    epsilon_f = 1.198 * df / (d - dc + 2 * dg)

    # 计算 σf
    sigma_f = E0 * epsilon_f * (1 + 0.5 * alpha_prime * epsilon_f)

    return sigma_f