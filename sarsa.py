import gym
import numpy as np
#from numpy.ma.core import shape
import pickle as pkl

cliffEnv=gym.make("CliffWalking-v0")

#inintializing action value function which is a table in this case
q_table =np.zeros(shape=(48,4)) #48 states and  4 colums which will hold the states for the Q value all initialized with zero


# explre is epsilon here and epsilon greedy policy  takes an action randomly with epsilon prob and  takes an optimum action with 1-epsilon prob
def policy(state, explore=0.0):
    action =int(np.argmax(q_table[state]))   #this is taking optimum action
    if np.random.random()<=explore:
        action = int(np.random.randint(low=0, high=4, size=1))  #this is taking random  action
    return action

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500

# Training for 500 episodes
for episode in range(NUM_EPISODES):
    done=False
    total_reward = 0
    episode_length = 0

    state = cliffEnv.reset()

    # Selecting an action according to our policy
    action = policy(state,EPSILON)

    # For every step of the episode
    while not done:
        # Take an action in the environment and observing reward and state
        next_state, reward, done, _ = cliffEnv.step(action)

        # Select the next action from the same policy
        next_action = policy(next_state, EPSILON)

        # SARSA update  just formula
        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        #making next stae as current state and next action as curr action
        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    print("Episode:", episode, "Episode Length:", episode_length, "Total Reward: ", total_reward)
cliffEnv.close()

pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved")