class Environment:
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def observationShape(self):
        raise NotImplementedError()

    def numActions(self):
        raise NotImplementedError()
