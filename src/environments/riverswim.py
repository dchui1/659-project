import numpy as np
from src.environments.Environment import Environment
from src.environments.Renderable import Renderable
import random

class RiverSwim(Environment, Renderable):
  steps = 0
  def __init__(self, params):
    Renderable.__init__(self)
    self.STEPS_LIMIT = params['steps'] # number of steps in episode (set to 5000)
    self.pos = 0
    self.swimRightStay = 0.6
    self.swimRightUp = 0.3
    self.swimRightDown = 0.1

    self.S1swimRightStay = 0.7
    self.S1swimRightUp = 0.3

    self.SNswimRightDown = 0.7
    self.SNswimRightStay = 0.3
    self.river_length = 5 # configurable?

  def start(self):
    self.pos = np.random.choice([1, 2])
    self.steps = 0
    print(self.pos)
    return np.array([self.pos])

  def step(self, a): # the transition function?
    old_pos = self.pos
    # if action is 0 then we do nothing
    if a == 1:
      # determine if we will successfully take the "up" action
      flip = random.random()
      if self.pos <= 0: # first state in chain
        if flip > self.S1swimRightUp:
          self.pos = self.pos + 1
      elif self.pos >= 5: # end of chain
        if flip <= self.SNswimRightDown:
          self.pos = self.pos - 1
      else: # middle of chain
        if flip <= self.swimRightDown:
          self.pos = self.pos - 1
        elif flip > self.swimRightDown + self.swimRightStay:
          self.pos = self.pos + 1
    elif a == 0: # make sure we always more to the left if we take action 0
      self.pos = self.pos - 1

    # make sure that the position we return (the next state) is between 0 and 5
    self.pos = np.clip(self.pos, 0, 5)

    done = self.STEPS_LIMIT == self.steps
    self.steps += 1

    # tuple indicating (state, reward, terminated, action)
    # note this is a continuing task, so the environment will only
    # terminate when max number of steps is reached
    return (self.rewardFunction(old_pos, a), np.array([self.pos]), done)

  def rewardFunction(self, x, a):
    if x >= self.river_length and a == 1:
      return 10000 # maybe change to 10
    if x <= 0 and a == 0:
      return 5.0
    return 0.0

  def observationShape(self):
    # position on the river.
    # states are: 0, 1, 2, 3, 4, 5
    return [6]

  def numActions(self):
    # (0) stay or (1) swim up the river
    return 2

  def render(self):

    from src.environments import rendering
    if self.viewer is None:
        print("Create viewer")
        self.viewer = rendering.Viewer(600,100)

    # create rendering geometry
    if self.render_geoms is None:
        # print("Create rendering geometry")
        # import rendering only if we need it (and don't import for headless machines)
        #from gym.envs.classic_control import rendering
        # from environments import rendering
        self.render_geoms = []
        self.render_geoms_xform = []
        # for entity in self.world.entities:
        geom = rendering.make_circle(1)
        xform = rendering.Transform()
        # if 'agent' in entity.name:
        geom.set_color(*self.AGENT_COLOR)
        # else:
        #     geom.set_color(*entity.color)
        geom.add_attr(xform)
        self.render_geoms.append(geom)
        self.render_geoms_xform.append(xform)

        goal1 = rendering.make_circle(1)
        goal1_xform = rendering.Transform()
        goal1.set_color(*self.GOAL_COLOR, alpha=0.5)
        goal1.add_attr(goal1_xform)
        self.render_geoms.append(goal1)
        self.render_geoms_xform.append(goal1_xform)

        goal2 = rendering.make_circle(1)
        goal2_xform = rendering.Transform()
        goal2.set_color(*self.GOAL_COLOR, alpha=0.5)
        goal2.add_attr(goal2_xform)
        self.render_geoms.append(goal2)
        self.render_geoms_xform.append(goal2_xform)
        # add geoms to viewer
        # for viewer in self.viewers:
        viewer = self.viewer
        viewer.geoms = []
        for geom in self.render_geoms:
            viewer.add_geom(geom)

    results = []

        # from multiagent import rendering
        # update bounds to center around agent

    cam_range = self.river_length


    pos = np.zeros(2)
    self.viewer.set_bounds(pos[0],pos[0]+cam_range,pos[1],1)
    # update geometry positions
    # for e, entity in enumerate(self.world.entities):
    #     self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
    print("New agent position:", self.pos, 1)

    self.render_geoms_xform[1].set_translation(self.river_length, 1) # big goal
    self.render_geoms_xform[2].set_translation(0, 1) #small goal
    self.render_geoms_xform[0].set_translation(self.pos, 1)

    # render to display or array
    results.append(self.viewer.render())

    return results
