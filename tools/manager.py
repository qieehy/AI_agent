import inspect
import json


class ToolManager:

    def __init__(self):
        self.tool_map = {}
        self.schemas = []

    def register(self, func):
        self.tool_map[func.__name__] = func
        schema = self.generate_schema(func)
        self.schemas.append(schema)
        return func

    def execute(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        func = self.tool_map[name]
        return func(**args)

    def generate_schema(self, func):
        signature = inspect.signature(func)
        properties = {}
        required = []

        for name, param in signature.parameters.items():
            properties[name] = {
                "type": self.python_type_to_json(param.annotation)
            }
            if param.default == inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",

            "function": {

                "name": func.__name__,

                "description":
                    func.__doc__ or "",

                "parameters": {

                    "type": "object",

                    "properties": properties,

                    "required": required

                }

            }
        }

    def python_type_to_json(self, py_type):
        mapping = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean"
        }
        return mapping.get(py_type, "string")



tool_manager = ToolManager()