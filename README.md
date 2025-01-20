# Cliff Walking Reinforcement Learning Project

This project implements and compares SARSA and Q-Learning algorithms in a Cliff Walking environment using OpenAI Gym. The agent must learn to navigate from a starting point to a goal while avoiding falling off a cliff, demonstrating fundamental concepts in reinforcement learning.

## Environment Description

The Cliff Walking environment consists of a 4x12 grid where:
- The agent (marked as 'A') starts from the bottom-left corner
- The goal (marked as 'G') is in the bottom-right corner
- The bottom edge (except for start and goal positions) represents a cliff (marked in pink)
- The agent must learn to navigate to the goal while avoiding the cliff
- Falling off the cliff results in a large negative reward and a reset to the start

Here's a visual representation of the environment:

```
```
![img.png](venv/img.png)
The agent can take four possible actions:
- Up (0)
- Right (1)
- Down (2)
- Left (3)

## Project Structure

```
cliff-walking/
├── cv_show.py          # Visualization utilities
├── evaluator.py        # Model evaluation script
├── qlearning.py        # Q-Learning implementation
├── random_agent.py     # Random agent baseline
├── sarsa.py           # SARSA implementation
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- gym==0.24.1
- numpy==1.23.0
- opencv-python==4.6.0.66
- cloudpickle==2.1.0
- pickle-mixin==1.0.2

## Implementation Details

### Algorithms

1. **SARSA (State-Action-Reward-State-Action)**
   - On-policy learning algorithm
   - Uses epsilon-greedy policy for exploration
   - Parameters:
     - EPSILON = 0.1 (exploration rate)
     - ALPHA = 0.1 (learning rate)
     - GAMMA = 0.9 (discount factor)

2. **Q-Learning**
   - Off-policy learning algorithm
   - Also uses epsilon-greedy policy
   - Same hyperparameters as SARSA

### Key Files

- `sarsa.py`: Implements the SARSA algorithm and saves the trained Q-table
- `qlearning.py`: Implements the Q-Learning algorithm
- `evaluator.py`: Loads trained models and visualizes their performance
- `random_agent.py`: Implements a baseline random agent for comparison
- `cv_show.py`: Contains visualization utilities using OpenCV

## Usage

1. **Train SARSA Agent**
```bash
python sarsa.py
```

2. **Train Q-Learning Agent**
```bash
python qlearning.py
```

3. **Evaluate Trained Agent**
```bash
python evaluator.py
```

4. **Run Random Agent Baseline**
```bash
python random_agent.py
```

## Visualization

The project includes a visualization component that shows:
- The grid environment (4x12 grid)
- Agent's position (marked as 'A')
- Cliff area (marked in pink with "Cliff" text)
- Goal position (marked as 'G')
- Grid lines for clear state separation

The visualization is implemented using OpenCV and updates in real-time as the agent moves through the environment.

## Training Parameters

- Number of episodes: 500
- Epsilon (exploration rate): 0.1
- Alpha (learning rate): 0.1
- Gamma (discount factor): 0.9

## Output

Both algorithms save their Q-tables in pickle format:
- `sarsa_q_table.pkl`: Contains the trained SARSA Q-table
- The evaluator can load these tables to visualize the learned policies

## Contributing

Feel free to fork this repository and submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for the Cliff Walking environment
- Reinforcement Learning principles based on Sutton and Barto's work