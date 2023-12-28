import pygame
import pygad
import pygad.gann
import pygad.nn
import random
import math
import numpy
MAP_WIDTH = 500
MAP_HEIGHT = 512
FPS = 60
DARK_GROUND = (124, 115, 46)


class Bird():
    def __init__(self):
        self.x = 80
        self.y = 250
        self.width = 34
        self.height = 24
        self.alive = True
        self.gravity = 0
        self.velocity = 0.3
        self.jump = -6

    def flap(self):
        self.gravity = self.jump

    def update(self):
        self.gravity += self.velocity
        self.y += self.gravity

    def is_dead(self, pipes):
        if self.y >= MAP_HEIGHT or self.y + self.height <= 0:
            return True
        for pipe in pipes:
            if not (
                self.x > pipe.x+pipe.width or
                self.x+self.width < pipe.x or
                self.y > pipe.y+pipe.height or
                self.y + self.height < pipe.y
            ):
                return True
        return False


class Pipe():
    def __init__(self, x=0, y=0, height=40):
        self.x = x
        self.y = y
        self.width = 40
        self.height = height
        self.speed = 3

    def update(self):
        self.x -= self.speed

    def is_out(self):
        if self.x+self.width <= 0:
            return True
        return False


class Game():
    def __init__(self):
        self.pipes = []
        self.birds = []
        self.score = 0
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.spawn_interval = 90
        self.interval = 0
        self.Neuro = Neuroevolution()
        self.gen = []
        self.alives = 0
        self.generation = 0
        self.background_speed = 0.5
        self.background_x = 0
        self.max_score = 0

    def start(self):
        self.interval = 0
        self.score = 0
        self.pipes = []
        self.birds = []
        self.Neuro.score_map.clear()
        self.gen = self.Neuro.ga_instance.population.copy()
        for i in self.gen:
            b = Bird()
            self.birds.append(b)
        self.generation += 1
        self.alives = len(self.birds)

    def update(self):
        next_holl = 0
        if self.alives > 0:
            for i in range(0, len(self.pipes), 2):
                if self.pipes[i].x + self.pipes[i].width > self.birds[0].x:
                    next_holl = self.pipes[i].height/self.height
                    break
        for bird in self.birds:
            if bird.alive is True:
                idx = self.birds.index(bird)
                solution = self.gen[idx]
                inputs = numpy.array(
                    [bird.y/bird.height, next_holl]).reshape(1, -1)
                pred = self.Neuro.predict(inputs, idx)
                if (pred[0] > 0.5):
                    bird.flap()

                bird.update()
                if bird.is_dead(self.pipes) is True:
                    bird.alive = False
                    self.alives -= 1
                    # self.Neuro.score_map[list2hash(solution)] = self.score
                    self.Neuro.score_map[idx] = self.score
                    if self.is_end() is True:

                        self.Neuro.ga_instance.run()

                        self.start()

        for i in range(len(self.pipes)-1, -1, -1):
            pipe = self.pipes[i]
            pipe.update()
            if pipe.is_out() is True:
                del self.pipes[i]

        if self.interval == 0:
            delta_board = 50
            pipe_hole = 120
            hole_position = round(
                random.random()*(self.height-delta_board*2-pipe_hole))+delta_board
            self.pipes.append(Pipe(x=self.width, y=0, height=hole_position))
            self.pipes.append(
                Pipe(x=self.width, y=hole_position+pipe_hole, height=self.height))

        self.interval += 1
        if self.interval == self.spawn_interval:
            self.interval = 0

        self.score += 1
        self.max_score = max(self.score, self.max_score)

    def is_end(self):
        if self.alives == 0:
            return True
        else:
            return False

    def display(self):
        global GAME_SPRITES, SCREEN, arial18
        background = GAME_SPRITES['background']
        pipetop = pygame.transform.rotate(GAME_SPRITES['pipe'], 180)
        pipebottom = GAME_SPRITES['pipe']
        bird_raw = GAME_SPRITES['bird']
        for i in range(math.ceil(self.width/background.get_width())+1):
            SCREEN.blit(background, (
                i*background.get_width()-math.floor(self.background_x % background.get_width()), 0))

        for pipe in self.pipes:
            idx = self.pipes.index(pipe)
            if idx % 2 == 0:
                SCREEN.blit(pipetop, pygame.Rect(
                    pipe.x, pipe.y+pipe.height-pipetop.get_height(), pipe.width, pipetop.get_height()))
            else:
                SCREEN.blit(pipebottom, pygame.Rect(
                    pipe.x, pipe.y, pipe.width, pipebottom.get_height()))
        for bird in self.birds:
            if bird.alive is True:
                bird_ro = pygame.transform.rotate(
                    bird_raw, math.pi/2*bird.gravity/20)
                SCREEN.blit(bird_ro, (bird.x, bird.y))
        text = f"Generation: {self.generation}\nMax Score:{self.max_score}\nScore:{self.score}\nAlive:{self.alives}/{self.Neuro.ga_instance.pop_size[0]}".split(
            '\n')
        for i, t in enumerate(text):
            text = arial18.render(t, True, DARK_GROUND)
            textX = text.get_rect().width
            textY = text.get_rect().height
            SCREEN.blit(text, (self.width/2-textX/2, (i*textY)))


class Neuroevolution():
    def __init__(self):
        self.GANN_instance = pygad.gann.GANN(num_solutions=50,
                                             num_neurons_input=2,
                                             num_neurons_hidden_layers=[2],
                                             num_neurons_output=1,
                                             hidden_activations=["sigmoid"],
                                             output_activation="sigmoid")
        self.population_vectors = pygad.gann.population_as_vectors(
            population_networks=self.GANN_instance.population_networks)
        self.initial_population = self.population_vectors.copy()
        self.ga_instance = pygad.GA(num_generations=1,
                                    num_parents_mating=10,
                                    initial_population=self.initial_population,
                                    sol_per_pop=50,
                                    keep_elitism=5,

                                    fitness_func=self.fitness_func,
                                    mutation_type="random",
                                    init_range_low=-2,
                                    init_range_high=5,
                                    keep_parents=2,
                                    on_generation=self.callback_generation
                                    )

        self.score_map = {}

    def fitness_func(self, ga_instance, solution, sol_idx):
        return self.score_map[sol_idx]

    def predict(self, data_inputs, idx):

        return pygad.nn.predict(last_layer=self.GANN_instance.population_networks[idx], data_inputs=data_inputs, problem_type="regression")

    def callback_generation(self, ga_instance):

        population_matrices = pygad.gann.population_as_matrices(
            population_networks=self.GANN_instance.population_networks, population_vectors=ga_instance.population)
        self.GANN_instance.update_population_trained_weights(
            population_trained_weights=population_matrices)


pygame.init()
SCREEN = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
pygame.display.set_caption("Flappy Bird GA")
clock = pygame.time.Clock()
arial18 = pygame.font.SysFont('arial', 18, False, False)
GAME_SPRITES = {}
GAME_SPRITES['pipe'] = pygame.image.load("./assets/pipe.png").convert_alpha()
GAME_SPRITES['background'] = pygame.image.load(
    "./assets/background.png").convert_alpha()
GAME_SPRITES['bird'] = pygame.image.load("./assets/bird.png").convert_alpha()

game = Game()
game.start()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    game.update()
    game.display()
    pygame.display.flip()
    clock.tick(FPS)
