class Memory():
    def __init__(self,message=None):
        self.messages = [message]


    def add_message(self,message):
        self.messages.append(message)


    def get_messages(self):
        return self.messages


