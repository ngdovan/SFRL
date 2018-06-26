import pygame
from game.env.CDEnv import CDEnv
from game.env.GridWorldEnv import GridWorldEnv
from game.agent.CDAgent import CDAgent
from game.agent.GridWorldAgent import GridWorldAgent
from game.agent.GridWorldMultipleAgent import GridWorldMultipleAgent
from game.env.target import Target
# import cv2


class Game:
    def __init__(self, trainingMode=False, rootFol='', visual=True, name='CDGame', matmap=None):
        pygame.init()
        pygame.font.init()
        self.matmap = matmap
        self.finish = False
        self.trainingMode = trainingMode
        self.visual = visual
        self.name = name

        SCREENWIDTH = 512
        SCREENHEIGHT = 512

        self.screen_size = (SCREENWIDTH, SCREENHEIGHT)
        if self.visual:
            self.screen = pygame.display.set_mode(self.screen_size)
        else:
            self.screen = pygame.display.set_mode((1, 1))
        self.background = pygame.Surface(self.screen_size)
        pygame.display.set_caption("DeepRL")
        self.clock = pygame.time.Clock()
        self.myfont = pygame.font.SysFont('Comic Sans MS', 16)
        self.rootFol = rootFol

    def new(self, randomPutOn=True, num_agents=2, num_target=1):
        self.finish = False
        if self.name == 'CDGame':
            self.env = CDEnv(self.rootFol)
        elif self.name == 'GridWorld':
            self.env = GridWorldEnv(self.rootFol, self.screen, matmap=self.matmap)

        self.env.all_agent = pygame.sprite.Group()
        self.env.all_sprites = pygame.sprite.Group()
        self.env.all_targets = pygame.sprite.Group()
        self.env.stones = pygame.sprite.Group()

        self.env.refresh(self.background)

        if self.name == 'CDGame':
            for i in range(num_agents):
                self.env.all_agent.add(CDAgent(self.env, self.trainingMode, self.rootFol, id=i))
        elif self.name == 'GridWorld':
            if num_agents <= 1:
                agent = GridWorldAgent(self.env, self.trainingMode, self.rootFol, myid=0)
                if randomPutOn:
                    self.env.random_put_on(agent)
                else:
                    self.env.put_on(agent, 1, 1)
                self.env.all_agent.add(agent)
            else:
                for i in range(num_agents):
                    agent = GridWorldMultipleAgent(self.env, self.trainingMode, self.rootFol, myid=i)
                    if randomPutOn:
                        self.env.random_put_on(agent)
                    else:
                        self.env.put_on(agent, 1, 1)
                    self.env.all_agent.add(agent)
            for i in range(num_target):
                target = Target(self.env, 0, 0)
                self.env.random_put_on(target)
                # self.env.put_on(target, 16, 16)
                self.env.all_targets.add(target)

        self.render()

    def get_1st_view(self, agentId):
        return self.getAgentById(agentId).observation(self.background)

    def getAgentById(self, Id):
        if len(self.env.all_agent) > 0:
            for index, spr in enumerate(self.env.all_agent):
                if spr.Id == Id:
                    return spr
        print('agent id={} not found!'.format(Id))
        return None

    def step(self, agentId, action):
        agent = self.getAgentById(agentId)
        if agent is None:
            print('agent {} not found!'.format(agentId))
        try:
            # if not agent.finish:
            hit, reward = agent.move(action+1)
            if self.trainingMode:
                self.update_screen()
            return agent.observation(self.background), reward, agent.finish, agent.reward
        except Exception as ex:
            print(action)
            print('agent pos: {},{}'.format(agent.rect.x, agent.rect.y))
            print(agent.targetsInfo)
            print(ex)

    def render(self):
        self.env.refresh(self.background)
        if self.visual:
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()

    def update_screen(self):
        if self.visual:
            self.screen.blit(self.background, (0, 0))
            self.env.all_sprites.draw(self.screen)
            self.env.all_targets.draw(self.screen)
            self.env.all_agent.draw(self.screen)
            for agent in self.env.all_agent:
                agent.updateReward(self.myfont, self.screen)
                agent.update_targets_info()
            pygame.display.flip()

    def run(self, random_move=False):
        self.render()
        self.update_screen()
        carryOn = True
        while carryOn:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    carryOn = False
            keys = pygame.key.get_pressed()
            for s in self.env.all_agent:
                if keys[pygame.K_LEFT]:
                    s.move(4)
                    self.update_screen()
                if keys[pygame.K_RIGHT]:
                    s.move(2)
                    self.update_screen()
                if keys[pygame.K_UP]:
                    s.move(1)
                    self.update_screen()
                if keys[pygame.K_DOWN]:
                    s.move(3)
                    self.update_screen()
            if random_move:
                for agent in self.env.all_agent:
                    if not self.trainingMode:
                        agent.random_walk()
                        self.update_screen()
                        self.clock.tick(60)
                        # agent.observation(self.background)
                        cv2.imshow('1st view', agent.firstView(self.background))
                        cv2.waitKey(1)
                    if agent.finish:
                        self.finish = True
            # self.update_screen()
            # self.clock.tick(60)
        pygame.quit()




