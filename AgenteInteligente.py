import pygame # type: ignore
import numpy as np
import random
import time

# Configurações
GRID_SIZE = 8
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
START = (0, 0)
GOAL = (7, 7)
OBSTACLES = [
    (0,1), (1,1), (2,1), (4,1), (4,2),
    (3,2), (3,4), (3,5), (4,5), (5,5),
    (5,4), (6,6), (7,6), (1,3), (2,4),
    (0,5), (2,6), (0,7)]

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # cima, baixo, esquerda, direita

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

def get_reward(state):
    if state == GOAL:
        return 100
    elif state in OBSTACLES:
        return -10
    else:
        return -1

def draw_grid(screen):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE
            if (x, y) in OBSTACLES:
                color = BLACK
            elif (x, y) == START:
                color = GREEN
            elif (x, y) == GOAL:
                color = RED
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)

def draw_agent(screen, pos, color=BLUE):
    rect = pygame.Rect(pos[0] * CELL_SIZE + 10, pos[1] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.ellipse(screen, color, rect)

def process_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# -----------------------------
# TREINAMENTO COM VISUALIZAÇÃO
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treinamento do Agente")
clock = pygame.time.Clock()

for episode in range(EPISODES):
    print(f"Treinando episódio {episode + 1}/{EPISODES}")
    state = START
    for step in range(MAX_STEPS):
        process_events()

        if random.random() < EPSILON:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        old_value = q_table[state[0], state[1], action]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] = old_value + ALPHA * (reward + GAMMA * next_max - old_value)

        state = next_state

        # Visualização do treinamento
        screen.fill(WHITE)
        draw_grid(screen)
        draw_agent(screen, state, ORANGE)
        pygame.display.flip()
        clock.tick(60)

        if state == GOAL:
            break

print("Treinamento concluído!")
time.sleep(1)
pygame.display.quit()

# -----------------------------
# EXECUÇÃO DO AGENTE TREINADO
# -----------------------------
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

    # Desenha caminho já percorrido
    for pos in path:
        draw_agent(screen, pos, BLUE)

    # Desenha agente atual
    if not reached_goal:
        draw_agent(screen, agent_pos, BLUE)

    pygame.display.flip()
    clock.tick(30)  # FPS alto para manter janela fluida

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
                time.sleep(0.8)  # <-- controle de velocidade da execução
        else:
            reached_goal = True
            print("\nCaminho percorrido pelo agente:")
            print(path)