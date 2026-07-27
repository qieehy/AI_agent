
class Agent:

    def __init__(self, memory, llm, tools, MAX_STEPS=100):
        self.MAX_STEPS = MAX_STEPS
        self.memory = memory
        self.llm = llm
        self.tools = tools

    def run(self, user_input):


        self.memory.add_message(
            {
                "role": "user",
                "content": user_input
            }
        )


        for _ in range(self.MAX_STEPS):

            response = self.llm.chat(self.memory.get_messages(), tools = self.tools.schemas)
            if response is None:
                return "llm 不可用"

            message = response.choices[0].message

            if message.tool_calls:
                self.memory.add_message(message)

                for tool_call in message.tool_calls:
                    result = self.tools.execute(tool_call)
                    self.memory.add_message({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })

            else:
                return message.content

        return f"reached {self.MAX_STEPS} tool calling steps"

