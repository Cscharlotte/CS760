import numpy as np

Q_tab = {'A': {'stay': 0, 'move': 0}, 'B': {'stay': 0, 'move': 0}}
alpha = 0.5
gamma = 0.8
epsilon = 0.5
current_state = 'A'
## deterministic greedy
for step in range(200):
    move_q = Q_tab[current_state]['move']
    stay_q = Q_tab[current_state]['stay']
    if move_q >= stay_q:
        next_action = 'move'
    else:
        next_action = 'stay'
    if next_action == 'stay':
        reward = 1
    else:
        reward = 0
    if next_action == 'move' and current_state == 'B':
        next_state = 'A'
    if next_action == 'move' and current_state == 'A':
        next_state = 'B'
    if next_action == 'stay':
        next_state = current_state
    Q_tab[current_state][next_action] = (1-alpha) * (Q_tab[current_state][next_action]) + alpha * (reward + gamma * max(Q_tab[next_state]['stay'], Q_tab[next_state]['move']))
    current_state = next_state

## epsilon-greedy
for step in range(200):
    move_q = Q_tab[current_state]['move']
    stay_q = Q_tab[current_state]['stay']
    if move_q >= stay_q:
        next_action = 'move'
    else:
        next_action = 'stay'
    if np.random.uniform(0, 1) < epsilon:
        next_action = np.random.choice(['move', 'stay'])
    if next_action == 'stay':
        reward = 1
    else:
        reward = 0
    if next_action == 'move' and current_state == 'B':
        next_state = 'A'
    elif next_action == 'move' and current_state == 'A':
        next_state = 'B'
    else:
        next_state = current_state
    Q_tab[current_state][next_action] = (1-alpha) * (Q_tab[current_state][next_action]) + \
                                          alpha * (reward + gamma * max(Q_tab[next_state]['stay'], Q_tab[next_state]['move']))
    current_state = next_state

