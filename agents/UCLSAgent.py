import torch
import numpy as np
import time
from agents.Agent import Agent

class UCLSAgent(Agent):
    def __init__(self, state_shape, num_acts, params):

        # self.config = config
        #
        # self.network = self.config.network
        #
        # self.state_dim = self.config.input_dims
        # self.num_actions = self.config.num_actions
        # self.linear_dim = self.config.network_dims[-2]

        self.tensorType = torch.DoubleTensor
        self.num_actions = num_acts
        self.linear_dim = np.prod(state_shape)
        self.state_dimensions = state_shape


        self.mem_size = self.linear_dim*self.num_actions

        self.p = params["p"]
        self.eta = params["eta"]
        self.beta = params["beta"]
        self.c_max = params["c_max"]
        self.gamma = params["gamma"]
        # self.greedy = self.config.greedy
        self.greedy = True
        self.thompson_beta = params["thompson_beta"]
        self.lambdaa = params["lambdaa"]
        self.lambdaa = 0

        self.use_retroactive = True


        self.ub_wt = np.sqrt(1.0+(1.0/self.p))

        self.Amat = np.zeros([self.mem_size, self.mem_size])
        self.bvec = np.zeros(self.mem_size)
        self.zvec = np.zeros(self.mem_size)
        self.weights = np.zeros(self.mem_size)

        self.Bmat = np.eye(self.mem_size)
        # self.Cmat = self.c_max*np.eye(self.mem_size)
        self.c_vec = self.c_max * np.ones(self.mem_size)
        # print("C matrix", self.Cmat)
        # print("c vec", self.c_vec)

        self.etaI = self.eta*np.eye(self.mem_size)
        self.cvec = self.c_max*np.ones(self.mem_size)
        self.nuvec = np.zeros(self.mem_size)

        self.temp_features = np.zeros(self.linear_dim)
        self.temp_representation = np.zeros(self.mem_size)
        self.temp_values = np.zeros(self.num_actions)
        self.temp_ub = np.zeros(self.num_actions)
        self.temp_mean = np.zeros(self.num_actions)

        self.current_state = np.zeros(self.state_dimensions).flatten()
        self.current_state_representation = np.zeros(self.mem_size)
        self.current_action = -1

    # def start_alt(self, state, action):
    #     self.zvec *= 0.0
    #     self.current_state[:] = state
    #     self.current_action = action

    def start(self, state):
        self.zvec *= 0.0

        self.current_action = self.policy(state)
        # print("State in start", state)
        self.current_state[:] = self.get_representation(state)

        return self.current_action

    def step_all(self,reward,td_error):
        # print(td_error)

        # print("Current state representation", self.current_state_representation)
        step_start = time.time()
        index = self.current_state_representation.nonzero()[0][0]
        # print(self.current_state_representation)
        # print("index", index)
        # self.zvec *= (self.gamma*self.lambdaa)
        bvec_start = time.time()
        self.zvec = self.current_state_representation

        self.bvec *= (1-self.beta)
        self.bvec[index] += (self.beta*self.zvec[index]*reward)
        print("Time for bvec update", time.time() - bvec_start)


        # print("temp representation", self.temp_representation)
        # print("zvec", self.zvec)
        outer_start = time.time()
        outer_prod= np.outer(self.zvec,self.temp_representation)
        print("Time for outer product", time.time() - outer_start)

        amat_start = time.time()
        self.Amat *= (1-self.beta)
        self.Amat += (self.beta*outer_prod)

        print("Time for amat update", time.time() - amat_start)
        # Since our current state representation is 1 hot, the square of the norm is just 1, so we can remove this term
        # alpha = 0.01/((np.square(np.linalg.norm(self.Amat))*np.square(np.linalg.norm(self.current_state_representation)))+1.0)
        alpha_start = time.time()
        alpha = 0.01/((np.square(np.linalg.norm(self.Amat)))+1.0)
        print("Time for alpha update", time.time() - alpha_start)

        # print("Bmat shape", self.Bmat.shape)
        # print("state", self.current_state_representation)
        # print("BMAT", self.Bmat)
        # bmat_dot = self.Bmat.dot(self.current_state_representation)
        outer__prod_start = time.time()
        bmat_dot = self.Bmat[:, index]

        # print("Bmat dot", bmat_dot)
        # print("new bmat_dot", self.Bmat[:, index])
        # print("bmat dot shape", bmat_dot.shape)
        # print("A transpose", self.Amat.transpose())

        a_transpose_dot = self.Amat.transpose() @ bmat_dot
        # print("A transpose dot", self.Amat.transpose().dot(bmat_dot))
        amat_dot = self.Amat @ (a_transpose_dot - self.current_state_representation)
        # print("Amat dot", amat_dot)
        # outer_prod_a_current = np.outer(amat_dot,self.current_state_representation)
        outer_prod_a_current = np.zeros([self.mem_size, self.mem_size])
        outer_prod_a_current[:, index] = amat_dot
        # print("outer prod", outer_prod_a_current)
        print("Outer prod a time", time.time() - outer__prod_start)

        bmat_start = time.time()
        self.Bmat -= alpha*(outer_prod_a_current)
        print("time for bmat update", time.time() - bmat_start)

        avec_start = time.time()
        self.nuvec *= (1-self.beta)
        # print("Zvec", self.zvec)
        self.nuvec += (self.beta*td_error*self.zvec)

        avec = self.Bmat.transpose() @ (self.nuvec)
        print("Time for avec update", time.time() - avec_start)

        retro_start = time.time()
        if self.use_retroactive:
            temp = self.c_max
            avec_square = np.multiply(avec,avec)
            if np.max(avec_square) > self.c_max:
                self.c_max = np.max(avec_square)
            if self.c_max != temp:
                for i in range(self.mem_size):
                    self.Cmat[i,i] += self.cvec[i]*(self.c_max-temp)

        print("time for retro update", time.time() - retro_start)

        # for i in range(self.mem_size):
        #     if self.zvec[i] <= self.beta:
        #         continue
        #     # self.cvec[i] *= (1-self.beta)
        #     for j in range(self.mem_size):
        #         if self.zvec[j] <= self.beta:
        #             continue
        #         self.Cmat[i,j] *= (1-self.beta)
        #         self.Cmat[i,j] += (self.beta*avec[i]*avec[j])
        # self.Cmat[index,index] *= (1-self.beta)
        # self.Cmat[index,index] += (self.beta*avec[index]*avec[index])
        self.c_vec *= (1-self.beta)
        self.c_vec[index] += (self.beta *avec[index]*avec[index])

        weight_start = time.time()
        x = self.bvec-self.Amat @ (self.weights)
        # Optimization, (B + eta) * x
        # since eta is a diagonal matrix, this is equivalent
        self.weights += 0.1*(self.Bmat.transpose() @ x  + self.eta * x)

        # print(self.weights)
        # print("Weight update: ", (self.Bmat.transpose()+self.etaI).dot(self.bvec-self.Amat.dot(self.weights)))
        # print("A mat", self.Amat)
        # print("B mat", self.Bmat)
        #
        print("Time for step all", time.time() - step_start)
        print("Time for weights update", time.time() - weight_start)



    # def step_alt(self, reward, state, done, action):
    #     if not done:
    #         self.populate_td_features(state,action)
    #     else:
    #         self.populate_td_features(state=None)
    #
    #     td_error = reward - self.get_value()
    #     self.step_all(reward, td_error)
    #
    #     if not done:
    #         self.current_state[:] = state
    #         self.current_action = action

    def update(self, state, sp, reward, action, done):

        if not done:
            start = time.time()
            next_action = self.policy(state)
            # print("Time to get policy", time.time() - start)
            self.populate_td_features(state,next_action)
        else:
            self.populate_td_features(state=None)

        td_error = reward - self.get_value()
        self.step_all(reward, td_error)

        # print("The state at update", state)

        if not done:
            self.current_state[:] = self.get_representation(state)
            self.current_action = next_action
            return self.current_action

    def populate_td_features(self, state=None, action=None):
        self.current_state_representation.fill(0)
        self.temp_representation.fill(0)

        _state = torch.from_numpy(self.current_state).type(self.tensorType)
        # features = self.network.get_representation(state).detach().data.numpy()
        features = _state.flatten()
        self.current_state_representation[(self.linear_dim*self.current_action):(self.linear_dim*(self.current_action+1))] = features
        self.temp_representation[(self.linear_dim*self.current_action):(self.linear_dim*(self.current_action+1))] = features
        # print("Setting current state representation", self.current_state_representation)
        # time.sleep(5)
        if state is not None:
            # _state = torch.from_numpy(state).type(self.config.FloatTensor)
            # features = self.network.get_representation(_state).detach().data.numpy()
            features = self.get_representation(state)
            # print("features", features)
            self.temp_representation[(self.linear_dim*action):(self.linear_dim*(action+1))] -= (self.gamma*features)

    def getAction(self, Obs):
        # print("The action", self.current_action)
        return self.current_action

    def get_value(self):
        return np.dot(self.weights, self.temp_representation)

    def get_greedy_action(self):
        # print(self.temp_values)
        maxPos = np.where(self.temp_values>=np.max(self.temp_values))[0]
        chosenPos = maxPos[np.random.randint(len(maxPos),size=1)[0]]
        return chosenPos

    def policy(self, state):
        # _state = torch.from_numpy(state).type(self.tensorType)
        # Picking a greedy action
        # features = self.get_representation(state).detach().data.numpy()
        features = self.get_representation(state)

        for i in range(self.num_actions):
            self.temp_representation.fill(0)
            self.temp_representation[(self.linear_dim*i):(self.linear_dim*(i+1))] = features

            # temp = np.dot(self.temp_representation, self.Cmat.dot(self.temp_representation))
            temp = self.c_vec.dot(self.temp_representation)
            # print("temp temp vec", temp, temp_vec)
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


    def get_representation(self, state):
        # print("The state in get representation", state)
        # print(state.shape)
        representation = np.zeros(self.state_dimensions)

        representation[tuple(state)] = 1
        # print("Representation", representation.flatten())
        return representation.flatten()
