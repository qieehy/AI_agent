from .manager import  tool_manager

@tool_manager.register
def add(a: int, b: int)->int:
    """计算两整数相加"""
    return a + b

@tool_manager.register
def multiply(a:int, b:int)->int:
    """计算两整数相乘"""
    return a * b



