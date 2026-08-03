from typing import Annotated

def add(a: Annotated[int, "第一个加数"], b: Annotated[int, "第二个加数"]) -> int:
    """计算两整数相加"""
    return a + b

def multiply(a: Annotated[int, "第一个乘数"], b: Annotated[int, "第二个乘数"]) -> int:
    """计算两整数相乘"""
    return a * b


TOOLS = [add, multiply]