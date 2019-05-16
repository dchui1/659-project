import random
import math
import numpy as np

from src.environments.riverswim import RiverSwim

params = {
    "steps": 5000,
    "episodes": 1
    }
env = RiverSwim(params)
reward_at_initial_state = 5.0
reward_at_final_state = 10000.0
reward_at_all_other_states = 0.0
np.random.seed(42)


def check_move_left():
    env.pos = 5
    initial_pos = env.pos
    a = 0
    for i in range(initial_pos, -1, -1):
        print("Initial position: ", env.pos)
        ([new_pos], r, done, a) = env.step(a)
        print("New position: ", new_pos)
        print("reward = ", r)
        print("")
        if initial_pos >= 1:
            assert new_pos == (initial_pos - 1)
            assert r == reward_at_all_other_states
        else:
            assert new_pos == initial_pos == 0
            assert r == reward_at_initial_state
        initial_pos = env.pos
check_move_left()


def check_move_right():
    env.pos = 0
    initial_pos = env.pos
    a = 1
    count = 0
    for i in range(1000):
        ([new_pos], r, done, a) = env.step(a)
        if initial_pos <= 4:
            if new_pos == initial_pos + 1:
                count += 1
        else:
            if new_pos == initial_pos:
                assert(r == reward_at_final_state)
                count += 1
        initial_pos = env.pos

    percentage_time_moved_right = ((count/1000) * 100)
    print("Percentage of time that we took the right action and moved right = ", percentage_time_moved_right, "%")
    return
check_move_right()
