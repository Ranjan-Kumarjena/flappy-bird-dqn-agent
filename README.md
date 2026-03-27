# Flappy Bird AI using Deep Q-Network (DQN)
## project overview
This Project is an Implementation of Reinforcement Learning on the Flappy Bird game using **Deep Q-Network(DQN)**.
The agent learn how to play the game by interacting with the envornment ,taking action ,receiving rewards,and imporoving its policy over time.
The main objective the project is **maximize the total Reward** nad make the bird survive as long as possible without hitting pipes or ground .
## objective 
## Objective
The goal of the agent is to:

- learn the best action at each state
- avoid obstacles
- survive longer
- maximize cumulative reward
In simple words, the agent tries to play Flappy Bird better and better by learning from experience.
## Algorithm Used
This project uses **Deep Q-Network (DQN)**, which is a combination of:

- **Reinforcement Learning**
- **Deep Learning**
- DQN uses a neural network to predict Q-values for each possible action and helps the agent choose the best action.
## How It Works
The agent follows this learning process:

1. Observe the current state from the game
2. Choose an action:
   - flap
   - do nothing
3. Perform the action in the environment
4. Receive reward
5. Store experience `(state, action, reward, next_state, done)`
6. Sample past experiences from replay memory
7. Train the neural network
8. Gradually improve performance
## Reward Strategy
The reward function is designed to help the agent learn good behavior.

Example:

- **Positive reward** for staying alive
- **Positive reward** for passing pipes
- **Negative reward** for collision / game over
- ## Features
- Flappy Bird game environment
- DQN-based agent
- Experience Replay
- Epsilon-Greedy exploration
- Neural network for Q-value prediction
- Training and testing modes
- Reward-based learning
