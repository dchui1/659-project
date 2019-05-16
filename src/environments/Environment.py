from src.RLGlue.BaseEnvironment import BaseEnvironment

class Environment(BaseEnvironment):
    def start(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def observationShape(self):
        raise NotImplementedError()

    def numActions(self):
        raise NotImplementedError()
