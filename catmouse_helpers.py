# ********
# This file is individualized for NetID adsane.
# ********

import numpy as np

def get_discount_factor():
    """
    Returns: the discount factor used by the agent
    """
    ɣ = 0.5 # numbers closer to 1 put more emphasis on future rewards
    return ɣ

def choose_action(t, Qi, Ni):
    """
    Inputs:
    t: the current time-step
    Qi[k]: the current estimate for Q*(s_i, a_k)
    Ni[k]: the number of times a_k has been performed in s_i so far
    Returns:
    the index of the action chosen at time t
    """
    #explore = (np.random.rand() < max(1, Ni.sum())**(-1))
    #k = np.random.randint(len(Qi)) if explore else Qi.argmax()
    #return k
    # Strategy: Epsilon-Greedy
    epsilon = 0.5  # Exploration rate, adjust as needed

    #Explore with probability epsilon
    if np.random.rand() < epsilon:
        k = np.random.randint(len(Qi))
    else:
        k = Qi.argmax()  # Exploit by choosing the action with the maximum Q-value estimate

    return k

def choose_learning_rate(t, k_t, Qi, Ni, Qj, Nj):
    """
    Inputs:
    t: the current time-step
    k_t: the action index that was chosen at time t
    Qi[k]: the current estimate for Q*(s_i, a_k), where s_i is the state at time t
    Ni[k]: the number of times a_k has been performed in s_i so far
    Qj[k]: the current estimate for Q*(s_j, a_k), where s_j is the state at time t+1
    Nj[k]: the number of times a_k has been performed in s_j so far
    Returns:
    the learning rate used for the transition from s_i to s_j
    """

    # Strategy from lecture:
    # Inversely proportional to number of times action was chosen
    # Ni[k_t] > 0 since the TD update happens after the action has been taken
    # You can replace this with your own strategy
    epsilon = 0.5  # Exploration rate, adjust as needed

    # Explore with probability epsilon
    if np.random.rand() < epsilon:
        α = 1.0  # Use a fixed learning rate for exploration
    else:
        α = 1.0 / Ni[k_t]  # Use the inverse of the number of times action was chosen for exploitation

    return α
