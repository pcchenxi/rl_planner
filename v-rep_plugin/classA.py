
class Simu_env:
    def __init__(self, port_num):
        self.action_space = ['l', 'f', 'r', 'h', 'e']
        self.n_actions = len(self.action_space)
        self.port_num = port_num
        self.reached_index = -1

