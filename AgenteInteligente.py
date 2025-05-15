import pygame  # type: ignore
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 8
CELL_SIZE = 80
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
GOAL = (7, 7)

OBSTACLES = [
    (0, 1), (1, 1), (2, 1), (4, 1), 
    (4, 2), (3, 4), (4, 5), (5, 5), 
    (6, 1), (7, 3), (5, 4), (6, 6), 
    (7, 6), (1, 3), (2, 4), (0, 5), (2, 6)
]

TRAPS = [(0, 7), (2, 3), (3, 5), (4, 4), (5, 1), (7, 5)] 

TELEPORTATION = [(0, 6), (7, 2)]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Cima, baixo, esquerda, direita

# Parâmetros do Q-Learning
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
EPISODES = 300
MAX_STEPS = 100

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (200, 200, 200)
BLUE = (50, 50, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)

# Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Funções auxiliares
def is_valid(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_state(state, action):
    dx, dy = ACTIONS[action]
    next_state = (state[0] + dx, state[1] + dy)
    return next_state if is_valid(next_state) else state

def get_reward(state, steps):
    if state == GOAL:
        return 100 - steps  # quanto menos passos, maior vai ser a recompensa
    elif state in TRAPS:
        return -20
    elif state in OBSTACLES:
        return -10
    else:
        return -1

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            screen.blit(sky_img, (x * CELL_SIZE, y * CELL_SIZE))
            
            if (x, y) in OBSTACLES:
                screen.blit(obstacles_img, (x * CELL_SIZE, y * CELL_SIZE))
            elif (x, y) in TRAPS:
                screen.blit(trap_img, (x * CELL_SIZE, y * CELL_SIZE))
            elif (x, y) in TELEPORTATION:
                screen.blit(teleportation_img, (x * CELL_SIZE, y * CELL_SIZE))
            elif (x, y) == START:
                pygame.draw.rect(screen, GREEN, rect)
            elif (x, y) == GOAL:
                screen.blit(goal_img, (x * CELL_SIZE, y * CELL_SIZE))

            pygame.draw.rect(screen, GREY, rect, 1)

def draw_agent(screen, pos, img):
    screen.blit(img, (pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10))

def draw_text(screen, text, pos, font, color=BLACK, bg_color=(255, 255, 255, 200)):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(topleft=pos)

    background_rect = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
    background_rect.fill(bg_color) 
    screen.blit(background_rect, pos)

    screen.blit(text_surface, pos)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -----------------------------
# TREINAMENTO COM VISUALIZAÇÃO
# -----------------------------

pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)
obstacles_img = pygame.image.load("Bloco.png")
obstacles_img = pygame.transform.scale(obstacles_img, (CELL_SIZE, CELL_SIZE))
trap_img = pygame.image.load("Flower.png")
trap_img = pygame.transform.scale(trap_img, (CELL_SIZE, CELL_SIZE))
agent_img = pygame.image.load("Mario.png")
agent_img = pygame.transform.scale(agent_img, (CELL_SIZE - 10, CELL_SIZE - 10))
goal_img = pygame.image.load("Princesa.png")
goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
teleportation_img = pygame.image.load("Tunel.png")
teleportation_img = pygame.transform.scale(teleportation_img, (CELL_SIZE, CELL_SIZE))
sky_img = pygame.image.load("Céu.png")
sky_img = pygame.transform.scale(sky_img, (CELL_SIZE, CELL_SIZE))

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    print(f"Treinando episódio {episode + 1}/{EPISODES}")
    state = START
    steps = 0
    for step in range(MAX_STEPS):
        process_events()

        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state = get_next_state(state, action)
        steps += 1
        reward = get_reward(next_state, steps)

        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = next_state

        if state == (7, 2):
            state = (0,6)
        elif state == (0,6):
            state = (7, 2)

        screen.fill(WHITE)
        draw_grid(screen)
        draw_agent(screen, state, img=agent_img)

        pygame.display.flip()
        clock.tick(45)

        if state == GOAL:
            break

    print(f"Episódio {episode + 1} finalizado em {steps} passos. Pontuação: {get_reward(state, steps)}")

print("Treinamento concluído!")
time.sleep(1)
pygame.display.quit()

pygame.display.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Execução do Agente Treinado")
clock = pygame.time.Clock()

agent_pos = START
path = [agent_pos]
reached_goal = False
running = True

while running:
    process_events()

    screen.fill(WHITE)
    draw_grid(screen)

    for pos in path:
        draw_agent(screen, pos, img=agent_img)

    if not reached_goal:
        draw_agent(screen, agent_pos, img=agent_img)

    pygame.display.flip()
    clock.tick(30)

    if not reached_goal:
        if agent_pos != GOAL:
            action = np.argmax(q_table[agent_pos[0], agent_pos[1]])
            next_pos = get_next_state(agent_pos, action)
            if next_pos == agent_pos:
                print("Agente está preso!")
                reached_goal = True
            else:
                path.append(next_pos)
                agent_pos = next_pos

                if agent_pos == (7, 2):
                    agent_pos = (0, 6)
                    path.append(agent_pos)
                elif agent_pos == (0, 6):
                    agent_pos = (7, 2)
                    path.append(agent_pos)

                time.sleep(0.8)  # <-- controle de velocidade da execução
        else:
            reached_goal = True
            print("\nCaminho percorrido pelo agente:")
            print(path)