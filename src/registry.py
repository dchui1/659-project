# agents
from src.agents.TabularQ import TabularQ
# from src.agents.LinearQ import LinearQ
from src.agents.TabularRTabularQ import TabularRTabularQ
# from src.agents.BnnRTabularQ import BnnRTabularQ
from src.agents.BayesianQLearningTabular import BayesianQLearningTabular

from src.agents.Dual_Vf import Dual_Vf

from src.agents.Mixing_Agent import Mixing_Agent

from src.agents.RiverswimOptimal import Optimal
from src.agents.UCB import UCB
# from src.agents.LinearQ_TDistR import TDistRLinearQ
from src.agents.UCLSAgent import UCLSAgent

# environments
from src.environments.gridworld import GridWorld
from src.environments.riverswim import RiverSwim
from src.environments.ContinuousGridworld import CtsGridWorld

def getAgent(exp):
    if exp.agent == 'linear-q':
        # return LinearQ
        pass
    if exp.agent == 'tabular-q':
        return TabularQ
    if exp.agent == 'ucb':
        return UCB
    if exp.agent == 'tabular-r tabular-q':
        return TabularRTabularQ
    if exp.agent == 'bayesian-q-learning':
        return BayesianQLearningTabular

    if exp.agent == 'blr tabular-q':
        # return BnnRTabularQ
        pass
    if exp.agent == 'T-distribution-r linear-q':
        # return TDistRLinearQ
        pass

    if exp.agent == 'ucls tabular-q':
        return UCLSAgent

    if exp.agent == 'dual_vf':
        return Dual_Vf

    if exp.agent == 'mixing_agent':
        return Mixing_Agent

    raise NotImplementedError

def getEnvironment(exp):
    if exp.environment == 'cts-gridworld':
        return CtsGridWorld
    if exp.environment == 'gridworld':
        return GridWorld
    if exp.environment == 'riverswim':
        return RiverSwim


    raise NotImplementedError
