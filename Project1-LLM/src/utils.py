KNOWLEDGE_BASE = "knowledge_base"
TEMP_FOLDER = "temp"

formulas2code = r'''
Objective: Refer to the following example to generate one python function based on description.
Requirements:
- All calculations must be implemented in only one python function, not two or more.
- Only generate one function written in python language, do not provide unnecessary explanations.
- All input parameters in the python function default to None, and the default values for constant parameters are provided according to the Description.
- Adds a null check for the input parameters and returns an error if any of it is null.


######################
-Example1-
######################

## Description
闪络时导线上产生的过电压的公式：

\[ U = 30 \times k \times \left( \frac{h}{d} \right) \times I \]

其中：
- \( U \) - 导线上产生的过电压
- \( I \) - 雷电流
- \( h \) - 导线离地高度
- \( k \) - 系数，取决于雷电流反击的速率
- \( d \) - 发生闪络点到导线距离


## Python Function
```python
def calculate_overvoltage(I=None, h=None, d=None, k=None):
    """
    闪络时计算导线上产生的过电压

    参数：
    I : 雷电流 (单位：安培)，默认值为 None
    h : 导线离地高度 (单位：米)，默认值为 None
    d : 闪络点到导线的距离 (单位：米)，默认值为 None
    k : 系数，取决于雷电流反击的速率，默认值为 None

    返回：导线上产生的过电压 (单位：伏特)，如果任何输入为 None，则返回 None
    """
    if I is None:
        return ValueError("雷电流,不可以为空值。")
    elif h is None:
        return ValueError("导线离地高度,不可以为空值。")
    elif d is None:
        return ValueError("闪络点到导线的距离,不可以为空值。")
    elif k is None:
        return ValueError("系数(取决于雷电流反击的速率),不可以为空值。")
    # 计算过电压
    U = 30 * k * (h / d) * I
    return U
```
'''

code2json = r'''
Objective: Refer to the following example to generate Tool based on Function.
Requirements:
- Only generate Tool, do not provide unnecessary explanations.
- Generate corresponding "parameter" and "parameter_unit" fields for each Function parameter except that there is no unit. 
- Return in JSON format.

######################
-Example-
######################

# Python Function
```python
def calculate_overvoltage(I=None, h=None, d=None, k=9.18):
    """
    闪络时计算导线上产生的过电压

    参数：
    I : 雷电流 (单位：安培)，默认值为 None
    h : 导线离地高度 (单位：米)，默认值为 None
    d : 闪络点到导线的距离 (单位：米)，默认值为 None
    k : 系数，取决于雷电流反击的速率，默认值为 None

    返回：导线上产生的过电压 (单位：伏特)，如果任何输入为 None，则返回 None
    """
    if I is None:
        return ValueError("雷电流,不可以为空值。")
    elif h is None:
        return ValueError("导线离地高度,不可以为空值。")
    elif d is None:
        return ValueError("闪络点到导线的距离,不可以为空值。")
    elif k is None:
        return ValueError("系数(取决于雷电流反击的速率),不可以为空值。")
    else:
        # 计算过电压
        U = 30 * k * (h / d) * I
        return U
```

# Json Tool
```json
{
    "name": "calculate_overvoltage",
    "description": "闪络时计算导线上产生的过电压",
    "arguments": {
        "type": "object",
        "properties": {
            "I": {
                "type": "float",
                "description": "雷电流 (单位：安培)，默认值为 None",
            },
            "h": {
                "type": "float",
                "description": "导线离地高度 (单位：米)，默认值为 None",
            },
            "d": {
                "type": "float",
                "description": "闪络点到导线的距离 (单位：米)，默认值为 None",
            },
            "k": {
                "type": "float",
                "description": "系数，取决于雷电流反击的速率，默认值为 9.18",
            }
        },
        "required": ["I", "h", "d", "k"]
    }
}
```
'''

latex_delimiters = [
    {
        "right": "$$",
        "left": "$$",
        "display": True
    },
    {
        "right": "$",
        "left": "$",
        "display": False
    }
]

css = """
#progress-bar { margin: 20px 0; }
.file-preview { max-height: 300px; overflow-y: auto; }
.chatbot { min-height: 400px; }
"""

