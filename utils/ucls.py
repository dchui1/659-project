import torch
import numpy as np

class UCLS(object):
    def __init__(self, config):

        self.config = config

        self.network = self.config.network

        self.state_dim = self.config.input_dims
        self.num_actions = self.config.num_actions
        self.linear_dim = self.config.network_dims[-2]

        self.mem_size = self.linear_dim*self.num_actions

        self.p = self.config.p
        self.eta = self.config.eta
        self.beta = self.config.beta
        self.c_max = self.config.c_max
        self.gamma = self.config.gamma
        self.greedy = self.config.greedy
        self.thompson_beta = self.config.thompson_beta
        self.lambdaa = self.config.lambdaa
        self.use_retroactive = self.config.use_retroactive

        self.ub_wt = np.sqrt(1.0+(1.0/self.p))

        self.Amat = np.zeros([self.mem_size, self.mem_size])
        self.bvec = np.zeros(self.mem_size)
        self.zvec = np.zeros(self.mem_size)
        self.weights = np.zeros(self.mem_size)

        self.Bmat = np.eye(self.mem_size)
        self.Cmat = self.c_max*np.eye(self.mem_size)
        self.etaI = self.eta*np.eye(self.mem_size)
        self.cvec = self.c_max*np.ones(self.mem_size)
        self.nuvec = np.zeros(self.mem_size)

        self.temp_features = np.zeros(self.linear_dim)
        self.temp_representation = np.zeros(self.mem_size)
        self.temp_values = np.zeros(self.num_actions)
        self.temp_ub = np.zeros(self.num_actions)
        self.temp_mean = np.zeros(self.num_actions)

        self.current_state = np.zeros(self.state_dim)
        self.current_state_representation = np.zeros(self.mem_size)
        self.current_action = -1

    def start_alt(self, state, action):
        self.zvec *= 0.0
        self.current_state[:] = state
        self.current_action = action

    def start(self, state):
        self.zvec *= 0.0
        self.current_action = self.policy(state)
        self.current_state[:] = state
        return self.current_action

    def step_all(self,reward,td_error):
        # print(td_error)
        self.zvec *= (self.gamma*self.lambdaa)
        self.zvec += self.current_state_representation

        self.bvec *= (1-self.beta)
        self.bvec += (self.beta*self.zvec*reward)

        self.Amat *= (1-self.beta)
        self.Amat += (self.beta*np.outer(self.zvec,self.temp_representation))

        alpha = 0.01/((np.square(np.linalg.norm(self.Amat))*np.square(np.linalg.norm(self.current_state_representation)))+1.0)
        self.Bmat -= alpha*(np.outer(self.Amat.dot(self.Amat.transpose().dot(self.Bmat.dot(self.current_state_representation)) - self.current_state_representation),self.current_state_representation))

        self.nuvec *= (1-self.beta)
        self.nuvec += (self.beta*td_error*self.zvec)

        avec = self.Bmat.transpose().dot(self.nuvec)

        if self.use_retroactive:
            temp = self.c_max
            avec_square = np.multiply(avec,avec)
            if np.max(avec_square) > self.c_max:
                self.c_max = np.max(avec_square)
            if self.c_max != temp:
                for i in range(self.mem_size):
                    self.Cmat[i,i] += self.cvec[i]*(self.c_max-temp)

        for i in range(self.mem_size):
            if self.zvec[i] <= self.beta:
                continue
            # self.cvec[i] *= (1-self.beta)
            for j in range(self.mem_size):
                if self.zvec[j] <= self.beta:
                    continue
                self.Cmat[i,j] *= (1-self.beta)
                self.Cmat[i,j] += (self.beta*avec[i]*avec[j])

        self.weights += 0.1*((self.Bmat.transpose()+self.etaI).dot(self.bvec-self.Amat.dot(self.weights)))
        # print(self.weights)

    def step_alt(self, reward, state, done, action):
        if not done:
            self.populate_td_features(state,action)
        else:
            self.populate_td_features(state=None)

        td_error = reward - self.get_value()
        self.step_all(reward, td_error)

        if not done:
            self.current_state[:] = state
            self.current_action = action

    def step(self, reward, state, done):

        if not done:
            next_action = self.policy(state)
            self.populate_td_features(state,next_action)
        else:
            self.populate_td_features(state=None)

        td_error = reward - self.get_value()
        self.step_all(reward, td_error)

        if not done:
            self.current_state[:] = state
            self.current_action = next_action
            return self.current_action

    def populate_td_features(self, state=None, action=None):
        self.current_state_representation.fill(0)
        self.temp_representation.fill(0)

        _state = torch.from_numpy(self.current_state).type(self.config.FloatTensor)
        features = self.network.get_representation(_state).detach().data.numpy()
        self.current_state_representation[(self.linear_dim*self.current_action):(self.linear_dim*(self.current_action+1))] = features
        self.temp_representation[(self.linear_dim*self.current_action):(self.linear_dim*(self.current_action+1))] = features

        if state is not None:
            _state = torch.from_numpy(state).type(self.config.FloatTensor)
            features = self.network.get_representation(_state).detach().data.numpy()
            self.temp_representation[(self.linear_dim*action):(self.linear_dim*(action+1))] -= (self.gamma*features)

    def get_value(self):
        return np.dot(self.weights, self.temp_representation)

    def get_greedy_action(self):
        # print(self.temp_values)
        maxPos = np.where(self.temp_values>=np.max(self.temp_values))[0]
        chosenPos = maxPos[np.random.randint(len(maxPos),size=1)[0]]
        return chosenPos

    def policy(self, state):
        _state = torch.from_numpy(state).type(self.config.FloatTensor)
        # Picking a greedy action
        features = self.network.get_representation(_state).detach().data.numpy()
        for i in range(self.num_actions):
            self.temp_representation.fill(0)
            self.temp_representation[(self.linear_dim*i):(self.linear_dim*(i+1))] = features

            temp = np.dot(self.temp_representation, self.Cmat.dot(self.temp_representation))
            if temp <= 0:
                self.temp_ub[i] = 0
            else:
                self.temp_ub[i] = np.sqrt(temp)

            self.temp_mean[i] = self.get_value()

        if self.greedy:
            self.temp_values[:] = self.temp_mean + (self.ub_wt*self.temp_ub)
        else:
            self.temp_values[:] = [(self.temp_mean[i] + (self.thompson_beta*np.random.normal()*self.temp_ub[i])) for i in range(0,self.num_actions)]

        return self.get_greedy_action()
