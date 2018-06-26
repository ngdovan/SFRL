from game.game import Game
from game import matgame
from constants import RAND_MAP
from game.env.MatWorldEnv import MatWorldEnv, Map


class myEnv:
    def __init__(self, visual=False, game='GridWorld'):
        self.visual = visual
        self.observation_space_shape = [32*5, 32*5, 3] if game == 'CDGame' else 5*5*4 + 2*1  # [32*5, 32*5, 3]
        self.action_space = [1, 2, 3, 4]
        if game == 'MatWorld':
            self.game = matgame.Game(randMap=RAND_MAP)
        else:
            self.game = Game(trainingMode=True, rootFol='', visual=self.visual, name=game, matmap=Map().maps[9])

    def reset(self, num_agents=1, num_targets=1):
        self.game.new(num_agents=num_agents, num_target=num_targets)
        # self.game.run()
        self.game.render()
        if num_agents == 1:
            return self.game.get_1st_view(0)
        else:
            result = []
            for i in range(num_agents):
                result.append(self.game.get_1st_view(i))
            return result

    def step(self, action, agent=None):
        if agent is None:
            return self.game.step(0, action)
        else:
            return self.game.step(agent.Id, action)

def make(visual=True, game='GridWorld'):
    return myEnv(visual, game)
