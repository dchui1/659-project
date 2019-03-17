from environments.Environment import Environment
import numpy as np
class CtsGridWorld(Environment):
    _x = 0
    _y = 0

    AGENT_COLOR = np.array([1, 0, 0])
    GOAL_COLOR = np.array([0, 1, 0])


    steps = 0
    noise = 0.01

    def __init__(self, params):
        self.maxSteps = params['steps']
        self.stepSize = params['stepsize']


        self.viewer = None
        self._reset_render()


    def getReward(self):
        if self._x + self.stepSize >= 1 and self._y + self.stepSize >= 1:
            return 1
        return 0

    def step(self, action):

        noise = np.random.normal(0, self.noise)
        if action == 0:
            self._x = bound(self._x + self.stepSize + noise, 0, 1)
        elif action == 1:
            self._y = bound(self._y + self.stepSize + noise, 0, 1)
        elif action == 2:
            self._x = bound(self._x - self.stepSize + noise, 0, 1)
        elif action == 3:
            self._y = bound(self._y - self.stepSize + noise, 0, 1)

        print("Position", self._x, self._y)
        r = self.getReward()
        self.steps += 1
        done = r == 1 or self.maxSteps == self.steps

        return (np.array([self._x, self._y]), r, done, action)

    def reset(self):
        self.steps = 0
        self._x = 0
        self._y = 0

        self._reset_render()

        return np.array([0, 0])

    def observationShape(self):
        return [1, 1]

    def numActions(self):
        return 4

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self):

        from environments import rendering
        if self.viewer is None:
            print("Create viewer")
            self.viewer = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # print("Create rendering geometry")
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            # from environments import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            # for entity in self.world.entities:
            geom = rendering.make_circle(0.05)
            xform = rendering.Transform()
            # if 'agent' in entity.name:
            geom.set_color(*self.AGENT_COLOR, alpha=0.5)
            # else:
            #     geom.set_color(*entity.color)
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            goal = rendering.make_circle(0.05)
            goal_xform = rendering.Transform()
            goal.set_color(*self.GOAL_COLOR)
            goal.add_attr(goal_xform)
            self.render_geoms.append(goal)
            self.render_geoms_xform.append(goal_xform)

            # add geoms to viewer
            # for viewer in self.viewers:
            viewer = self.viewer
            viewer.geoms = []
            for geom in self.render_geoms:
                viewer.add_geom(geom)

        results = []

            # from multiagent import rendering
            # update bounds to center around agent
        cam_range = 1

        pos = np.zeros(2)
        self.viewer.set_bounds(pos[0],pos[0]+cam_range,pos[1],pos[1]+cam_range)
        # update geometry positions
        # for e, entity in enumerate(self.world.entities):
        #     self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        print("New agent position:", self._x, self._y)
        self.render_geoms_xform[0].set_translation(self._x, self._y)
        self.render_geoms_xform[1].set_translation(1, 1)

        # render to display or array
        results.append(self.viewer.render())

        return results



def bound(x, min, max):
    b = max if x >= max else x
    b = 0 if b <= min else b
    return b
