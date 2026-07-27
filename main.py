from agent import Agent
from memory import Memory
from llm import LLMClient
from tools import tool_manager

agent = Agent(memory = Memory({"role": "system", "content": "你是一个具备tool_calling的agent"}), llm=LLMClient(), tools=tool_manager)

while True:
    q=input(">")
    if q=="exit":
        break
    print(
        agent.run(q)
    )