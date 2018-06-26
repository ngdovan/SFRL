from pytmx.util_pygame import load_pygame
import pygame

from game.agent.iagent import IAgent
from game.env.stone import Stone
from game.env.border import Border
from game.env.background import Background
from game.env.object import Object
from game.env.target import Target
from game.env.ienv import IEnv
from random import randint
from pytmx import TiledTileLayer

BACKGROUND = 0
BRICK = 2
TARGET = 3
textures = {
    BACKGROUND: pygame.image.load('game/env/map/resources/gridworld/background.png'),
    BRICK: pygame.image.load('game/env/map/resources/gridworld/brick.png'),
    TARGET: pygame.image.load('game/env/map/resources/gridworld/target.png')
}
TILESIZE = 32
MAPWIDTH = 16
MAPHEIGHT = 16


class GridWorldEnv(IEnv):
    def __init__(self, rootFol='', screen=None, matmap=None):
        self.RootFol = rootFol
        tiled_map = load_pygame(rootFol + 'game/env/map/resources/gridworld/gridworld.tmx', invert_y=True)
        assert isinstance(tiled_map, object)

        self.all_agent = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()
        self.all_targets = pygame.sprite.Group()
        self.stones = pygame.sprite.Group()
        self.gameMap = tiled_map

        self.sprite_width = self.gameMap.tilewidth
        self.sprite_height = self.gameMap.tileheight

        if matmap is not None:
            # for layer in tiled_map.visible_layers:

            backgroundLayer = tiled_map.get_layer_by_name('background')
            brickLayer = tiled_map.get_layer_by_name('brick')
            targetLayer = tiled_map.get_layer_by_name('target')
            for i in range(MAPHEIGHT):
                for j in range(MAPWIDTH):
                    if matmap[i][j] == BACKGROUND:
                        # sprite = Background(self, i * TILESIZE, j * TILESIZE, textures[BACKGROUND])
                        # self.all_sprites.add(sprite)
                        gid = backgroundLayer.data[i][j]
                        # tiled_map.images[gid] = textures[BACKGROUND]
                    elif matmap[i][j] == BRICK:
                        # sprite = Stone(self, i * TILESIZE, j * TILESIZE, textures[BRICK])
                        # self.all_sprites.add(sprite)
                        gid = brickLayer.data[i][j]
                        # tiled_map.images[gid] = textures[BRICK]
                    elif matmap[i][j] == TARGET:
                        # sprite = Target(self, i * TILESIZE, j * TILESIZE, textures[TARGET])
                        # self.all_targets.add(sprite)
                        gid = targetLayer.data[i][j]
                        # tiled_map.images[gid] = textures[TARGET]

        self.hit_target_reward = 200
        self.hit_obj_penalty = -1
        self.walk_energy = -0.01
        self.mapwidth = self.gameMap.tilewidth * self.gameMap.width
        self.mapheight = self.gameMap.tileheight * self.gameMap.height

        self.screen = screen

    def refresh(self, background):
        self.all_sprites = pygame.sprite.Group()
        # draw map data on screen
        for layer in self.gameMap.visible_layers:
            for x, y, gid, in layer:
                tile = self.gameMap.get_tile_image_by_gid(gid)
                # print(layer.name)
                if tile is not None:
                    if layer.name == 'brick':
                        # print(layer.name)
                        self.all_sprites.add(Stone(self, x, y, tile))
                    elif layer.name == 'background':
                        # print(layer.name)
                        self.all_sprites.add(Background(self, x, y, tile))
                    elif layer.name == 'object':
                        # print(layer.name)
                        self.all_sprites.add(Object(self, x, y, tile))
                    elif layer.name == 'target':
                        self.all_sprites.add(Target(self, x, y, tile))

                    background.blit(tile, (x * self.gameMap.tilewidth, y * self.gameMap.tileheight))

    def check_col(self, sprite):
        if self.all_sprites is not None:
            all_cols = pygame.sprite.spritecollide(sprite, self.all_sprites, False, pygame.sprite.collide_mask)
            all_cols = [item for item in all_cols if not isinstance(item, Background)]
        else:
            all_cols = []
        if self.all_targets is not None:
            all_cols_with_targets = pygame.sprite.spritecollide(sprite, self.all_targets, False,
                                                                pygame.sprite.collide_mask)
        else:
            all_cols_with_targets = []
        if all_cols is not None and all_cols_with_targets is not None:
            all_cols.extend(all_cols_with_targets)
        if all_cols is None or len(all_cols) == 0:
            all_cols = all_cols_with_targets
        if all_cols is None:
            all_cols = []
        if sprite.rect.x < 0 or sprite.rect.y < 0 or \
                sprite.rect.x > self.mapwidth - self.sprite_width or \
                sprite.rect.y > self.mapheight - self.sprite_height:
            return True, all_cols

        if not isinstance(sprite, Target) and all_cols is not None:
            for s in all_cols:
                if isinstance(s, Target):
                    if sprite.rect.x == s.rect.x and sprite.rect.y == s.rect.y:
                        sprite.finish = True
                    else:
                        all_cols.remove(s)
        all_cols_with_agents = pygame.sprite.spritecollide(sprite, self.all_agent, False, pygame.sprite.collide_mask)
        if len(all_cols_with_agents) > 0 and isinstance(sprite, IAgent):
            all_cols_with_agents = [item for item in all_cols_with_agents if item.Id != sprite.Id]
            if len(all_cols_with_agents) > 0:
                all_cols.extend(all_cols_with_agents)

        return len(all_cols) > 0, all_cols

    def random_put_on(self, sprite):
        ran_x = randint(0, self.gameMap.width - 1)
        ran_y = randint(0, self.gameMap.height - 1)
        sprite.rect.x = ran_x * self.sprite_width
        sprite.rect.y = ran_y * self.sprite_height
        coled, _ = self.check_col(sprite)
        while coled:  # and self.check_col(sprite)[0]:
            ran_x = randint(0, self.gameMap.width - 1)
            ran_y = randint(0, self.gameMap.height - 1)
            sprite.rect.x = ran_x * self.sprite_width
            sprite.rect.y = ran_y * self.sprite_height
            coled, _ = self.check_col(sprite)

    def put_on(self, sprite, row, col):
        sprite.rect.y = (row - 1) * self.sprite_height
        sprite.rect.x = (col - 1) * self.sprite_width
        coled, _ = self.check_col(sprite)
        return not coled
