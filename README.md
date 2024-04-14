# Cat-Mouse
Homework for CIS 667 - Intro to AI. 

A simple RL experiment setup simulates a scenario in which a mouse (agent) and a cat (environment) interact within a grid. The objective for the mouse is to stay as far from the cat as possible to maximize its reward, which is quantified by the Chebyshev distance between them. The environment is a rectangular grid where the positions of the mouse and the cat are determined by their respective coordinates. 


1)catmouse_helpers File

1. **`get_discount_factor()`**:
   - This function returns a discount factor (denoted as \(\gamma\)), which is used to prioritize immediate rewards over future rewards in the calculation of the expected return. A value of 0.5 suggests that future rewards are considered less valuable than immediate ones by half with each step into the future.

2. **`choose_action(t, Qi, Ni)`**:
   - Inputs:
     - `t`: Current time-step.
     - `Qi`: Array of current Q-values for a specific state and all possible actions.
     - `Ni`: Array of counts of how many times each action has been taken in the current state.
   - The function decides whether to explore or exploit based on a fixed exploration rate (\(\epsilon\)). It chooses a random action with probability \(\epsilon\) (exploration) and the best-known action (highest Q-value) otherwise (exploitation). This approach is known as Epsilon-Greedy.

3. **`choose_learning_rate(t, k_t, Qi, Ni, Qj, Nj)`**:
   - Inputs:
     - `t`: Current time-step.
     - `k_t`: Index of the action chosen at time `t`.
     - `Qi`: Array of current Q-values for a specific state and all possible actions.
     - `Ni`: Array of counts of how many times each action has been taken in the current state.
     - `Qj`: Array of Q-values for the subsequent state and all possible actions.
     - `Nj`: Array of counts of how many times each action has been taken in the subsequent state.
   - The function returns a learning rate for the Q-value update. The learning rate can be fixed (for exploration) or inversely proportional to the number of times the chosen action has been taken in the current state (for exploitation), as recommended in some RL methodologies.

The catmouse_helpers script illustrates basic elements of a Q-learning agent, where decisions on actions and learning rates are made based on current estimates of states, actions, and their values. 

2)catmouse File

### State Representation and Action Space
- The `state` is represented by the coordinates of the mouse and cat `(mx, my, cx, cy)`.
- The mouse has 9 possible actions: moving in any direction or staying still, while the cat moves randomly.

### State-Index Mapping
- `state_to_index` and `index_to_state` functions convert between state tuples and unique indices, allowing the use of arrays for data storage.

### Reward Function
- The reward is based on the Chebyshev distance between the mouse and cat, encouraging the mouse to maximize this distance.

### Learning Components
- `Q` array: Stores estimated rewards for each state-action pair, updated based on the mouse's experiences.
- `visit_counts` and `choice_counts` arrays: Track how often each state and action have been visited or chosen, respectively.

### Simulation Loop
- The script runs a loop where the mouse selects actions, updates its state, and adjusts Q-values using a temporal-difference learning rule.

### Visualization
- The script visualizes the mouse's and cat's positions and the learning progress using matplotlib, showing improvements over time.
