class Agent:
    def __init_(self, state_shape, num_acts):
        self.state_shape = state_shape
        self.num_acts = num_acts

    # should take a state and return an action
    # also responsible for setting up any memory the agent may need
    # for instance initializing the value-function approximator
    def start(self, s):
        raise NotImplementedError

    # takes the standard (S, A, S', R) tuple
    # plus a flag indicating if the episode ended
    # should update the agent's value function and any other
    # internal information the agent may be keeping
    def update(self, s, sp, r, a, done):
        raise NotImplementedError

    # takes the current state and returns the action
    def getAction(self, s):
        raise NotImplementedError
