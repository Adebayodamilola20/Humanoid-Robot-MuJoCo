# Humanoid Reinforcement Learning with MuJoCo

This project implements and trains a Reinforcement Learning (RL) agent to control a humanoid character in a physics-based simulation. Using the **MuJoCo** physics engine and **Gymnasium** environment, the agent learns to walk and maintain balance through the **PPO** (Proximal Policy Optimization) algorithm.

## Features

- **Training**: Train a humanoid model using Stable-Baselines3.
- **Visualization**: Visualize the trained model or random actions in a 3D environment.
- **MyoSuite Integration**: Support for musculoskeletal simulations via MyoSuite.

## Project Structure

- `train_humanoid.py`: Script to initialize and train the PPO model on `Humanoid-v4`.
- `visualize_humanoid.py`: Script to load and visualize the trained `humanoid_brain.zip` model.
- `random_humanoid.py`: Simple script to see the humanoid performing random actions.
- `random_myosuite.py`: Demonstration of random actions in a MyoSuite environment.
- `humanoid_brain.zip`: Pre-trained model weights (if available).

## Installation

Ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install gymnasium[mujoco] stable-baselines3 myosuite
```

## Usage

### To Train:
```bash
python train_humanoid.py
```

### To Visualize:
```bash
python visualize_humanoid.py
```

## Credits
Project by **adebayodamilola20**
