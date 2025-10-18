# SARSA Learning Algorithm


## AIM
To implement SARSA Learning Algorithm.

## PROBLEM STATEMENT
The problem might involve teaching an agent to interact optimally with an environment (e.g., gym-walk), where the agent must learn to choose actions that maximize cumulative rewards using RL algorithms like SARSA and Value Iteration..

## SARSA LEARNING ALGORITHM
1.Initialize the Q-table, learning rate α, discount factor γ, exploration rate ϵ, and the number of episodes.
2.For each episode, start in an initial state s, and choose an action a using the ε-greedy policy.
3.Take action a, observe the reward r and the next state s′ , and choose the next action a′ using the ε-greedy policy.
4.Update the Q-value for the state-action pair (s,a) using the SARSA update rule.
5.Update the current state to s′ and the current action to a′.
6.Repeat steps 3-5 until the episode reaches a terminal state.
7.After each episode, decay the exploration rate 𝜖 and learning rate α, if using decay schedules.
8.Return the Q-table and the learned policy after completing all episodes.
## SARSA LEARNING FUNCTION
### Name:APARNA.M
### Register Number:212223220008

Include the SARSA Learning function

def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilon[e])
      while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = select_action(next_state, Q, epsilon[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

## OUTPUT:
<img width="719" height="263" alt="image" src="https://github.com/user-attachments/assets/bd91cfa8-eafb-4208-9a10-a66ebd416a67" />



<img width="867" height="648" alt="image" src="https://github.com/user-attachments/assets/c16cfaa2-3e85-415b-85e5-651d4ce5a796" />



<img width="562" height="266" alt="image" src="https://github.com/user-attachments/assets/944ed8e0-e79e-438e-976e-6abfac52e913" />





<img width="883" height="655" alt="image" src="https://github.com/user-attachments/assets/df0d8f5c-5e8d-452a-8c18-14101b0db138" />

## RESULT:
Thus, to implement SARSA learning algorithm is executed successfully.
