from src.RLGlue.BaseAgent import BaseAgent

# Takes an agent and converts its API to one that works with RLGlue
# keeps track of the previous (state, action) pair
# and passes that to the agent update function
class AgentWrapper(BaseAgent):
    def __init__(self, base_agent, params):
        self.agent = base_agent

        self.s_t = None
        self.a_t = None

    # Called at the beginning of each episode
    # Takes the initial state from environment
    def start(self, s):
        self.s_t = s
        self.a_t = self.agent.policy(s)
        return self.a_t

    # Called on each timestep (after the first)
    # updates the agent's value function and gets the next action
    def step(self, r_t, s_tp1):
        self.agent.update(self.s_t, s_tp1, r_t, self.a_t, False)

        self.s_t = s_tp1
        self.a_t = self.agent.policy(s_tp1)

        return self.a_t

    # Called whenever the agent transitions into a terminal state
    # There is no next state, so just pass a dummy to the agent
    def end(self, r):
        self.agent.update(self.s_t, self.s_t, r, self.a_t, True)
