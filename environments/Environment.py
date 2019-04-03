#from environments import rendering
class Environment:
    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def observationShape(self):
        raise NotImplementedError()

    def numActions(self):
        raise NotImplementedError()

    # def render(self, mode='human'):
    #     print("render in environment")
    #     # if mode == 'human':
    #     #     alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    #     #     message = ''
    #     #     for agent in self.world.agents:
    #     #         comm = []
    #     #         for other in self.world.agents:
    #     #             if other is agent: continue
    #     #             if np.all(other.state.c == 0):
    #     #                 word = '_'
    #     #             else:
    #     #                 word = alphabet[np.argmax(other.state.c)]
    #     #             message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
    #     #     print(message)
    #     #
    #     # for i in range(len(self.viewers)):
    #         # create viewers (if necessary)
    #         # import rendering only if we need it (and don't import for headless machines)
    #         #from gym.envs.classic_control import rendering
    #     self.viewer = rendering.Viewer(700,700)
    #
    #     # create rendering geometry
    #     if self.render_geoms is None:
    #         # import rendering only if we need it (and don't import for headless machines)
    #         #from gym.envs.classic_control import rendering
    #         from environments import rendering
    #         self.render_geoms = []
    #         self.render_geoms_xform = []
    #         for entity in self.world.entities:
    #             geom = rendering.make_circle(entity.size)
    #             xform = rendering.Transform()
    #             if 'agent' in entity.name:
    #                 geom.set_color(*entity.color, alpha=0.5)
    #             else:
    #                 geom.set_color(*entity.color)
    #             geom.add_attr(xform)
    #             self.render_geoms.append(geom)
    #             self.render_geoms_xform.append(xform)
    #
    #         # add geoms to viewer
    #         for viewer in self.viewers:
    #             viewer.geoms = []
    #             for geom in self.render_geoms:
    #                 viewer.add_geom(geom)
