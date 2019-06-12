import numpy as np

class Renderable():

    AGENT_COLOR = np.array([1, 0, 0])
    GOAL_COLOR = np.array([0, 1, 0])
    def __init__(self):
        print("Render init")
        self.viewer = None
        self._reset_render()

    #     self._x = 0
    #     self._y = 0


    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None


    def render(self):

        from src.environments import rendering
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
            geom = rendering.make_circle(1)
            xform = rendering.Transform()
            # if 'agent' in entity.name:
            geom.set_color(*self.AGENT_COLOR, alpha=0.5)
            # else:
            #     geom.set_color(*entity.color)
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            goal = rendering.make_circle(1)
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
        cam_range = self.shape[0]

        pos = np.zeros(2)
        self.viewer.set_bounds(pos[0],pos[0]+cam_range,pos[1],pos[1]+cam_range)
        # update geometry positions
        # for e, entity in enumerate(self.world.entities):
        #     self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
        print("New agent position:", self._x, self._y)
        self.render_geoms_xform[0].set_translation(self._x, self._y)
        self.render_geoms_xform[1].set_translation(self.shape[0], self.shape[1])

        # render to display or array
        results.append(self.viewer.render())

        return results
