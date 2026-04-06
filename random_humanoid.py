import gymnasium as gym
import time

def main():
    # Initialize the Humanoid-v4 environment with human render mode
    print("Initializing Humanoid-v4 environment...")
    env = gym.make("Humanoid-v4", render_mode="human")
    
    # Reset the environment to start
    observation, info = env.reset()
    
    # Run the loop for 10000 steps so it stays open longer
    for step in range(10000):
        # Add a tiny delay so it doesn't run through 1000 steps in 0.1 seconds
        time.sleep(0.01)
        
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        
        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # If the episode ends (e.g., the skeleton falls over and terminates), reset it
        if terminated or truncated:
            observation, info = env.reset()
            
    print("Finished 1000 steps.")
    env.close()

if __name__ == "__main__":
    main()
