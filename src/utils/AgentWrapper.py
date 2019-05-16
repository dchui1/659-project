from src.RLGlue.BaseAgent import BaseAgent

class AgentWrapper(BaseAgent):
    def __init__(self, base_agent):
        self.agent = base_agent

        self.s_t = None
        self.a_t = None

    def start(self, s):
        self.s_t = s
        self.a_t = self.agent.policy(s)
        return self.a_t

    def step(self, r_t, s_tp1):
        self.agent.update(self.s_t, s_tp1, r_t, self.a_t, False)

        self.s_t = s_tp1
        self.a_t = self.agent.policy(s_tp1)

        return self.a_t

    def end(self, r):
        self.agent.update(self.s_t, self.s_t, r, self.a_t, True)