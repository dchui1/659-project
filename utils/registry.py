# agents
from agents.TabularQ import TabularQ
from agents.LinearQ import LinearQ
from agents.TabularRTabularQ import TabularRTabularQ
from agents.BnnRTabularQ import BnnRTabularQ

from agents.RiverswimOptimal import Optimal
from agents.UCB import UCB
from agents.LinearQ_TDistR import TDistRLinearQ
from agents.BayesianQLearningTabular import BayesianQLearningTabular

# environments
from environments.gridworld import GridWorld
from environments.ContinuousGridworld import CtsGridWorld

def getAgent(exp):
    if exp.agent == 'linear-q':
        return LinearQ
    if exp.agent == 'tabular-q':
        return TabularQ
    if exp.agent == 'ucb':
        return UCB
    if exp.agent == 'tabular-r tabular-q':
        return TabularRTabularQ
    if exp.agent == 'bayesian-q-learning':
        return BayesianQLearningTabular

    if exp.agent == 'blr tabular-q':
        return BnnRTabularQ
    if exp.agent == 'T-distribution-r linear-q':
        return TDistRLinearQ

    raise NotImplementedError

def getEnvironment(exp):
    if exp.environment == 'cts-gridworld':
        return CtsGridWorld
    if exp.environment == 'gridworld':
        return GridWorld

    raise NotImplementedError
