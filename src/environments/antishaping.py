from src.environments.Environment import Environment
import numpy as np
class Antishaping(Environment):
    def __init__(self, params):

        self.num_states = params["size"]
        self.maxSteps = params["steps"]
        self.max_reward = params["max_reward"]
        self.shape_numerator = params["shape_numerator"]
        self.pos = 0
        self.previous_position = None

    def start(self):
        self.pos = 0
        self.previous_position = None
        self.steps = 0
        return np.array([self.pos])


    def step(self, action):
        self.previous_position = self.pos
        # 0 = left, 1 = right
        if action == 0:
            self.pos = bound(self.pos - 1, 0, self.num_states - 1)

        elif action == 1:
            self.pos = bound(self.pos + 1, 0, self.num_states - 1)

        r = self.getReward()
        self.steps += 1

        done = self.maxSteps == self.steps
        return (r, np.array([self.pos]), done)


    def observationShape(self):
        return [self.num_states]

    def numActions(self):
        return 2

    def getReward(self):
        left_reward = self.shape_numerator / (self.pos + 1)
        right_reward = self.shape_numerator / (self.pos + 1)
        if self.pos == self.num_states - 1:
            right_reward = self.max_reward
        if self.previous_position < self.pos:
            return right_reward
        else:
            return left_reward

def bound(x, min, max):
    b = max if x >= max else x
    b = 0 if b <= min else b
    return b

  #
  # void create_antishape(size_t num_states)
  # {
  #   total_reward = 0;
  #   num_steps = 0;
  #   horizon = num_states*2;
  #
  #   vector<state_reward> translation = make_translation(num_states);
  #
  #   start_state = translation[0].first;
  #   state = start_state;
  #   dynamics.resize(num_states);
  #   for (size_t i = 0; i < num_states; i++)
  #     {
	# uint32_t left_state = i==0? 0 : i-1;
	# uint32_t right_state = min(i+1,num_states-1);
  #
	# float left_reward = 0.25f / (float) (left_state+1);
	# float right_reward = 0.25f / (float) (right_state+1);
	# if (right_state == num_states-1)
	#   right_reward = 1.f;
  #
	# state_reward sr_left(translation[left_state].first, left_reward);
	# state_reward sr_right(translation[right_state].first, right_reward);
  #
	# dynamics[translation[i].first] = pair<state_reward,state_reward>(sr_left, sr_right);
  #     }
  # }
