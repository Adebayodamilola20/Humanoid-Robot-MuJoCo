import gymnasium as gym
from stable_baselines3 import PPO

def main():
    print("Initializing Humanoid-v4 environment with human render mode...")
    # Re-initialize the environment with human rendering
    env = gym.make("Humanoid-v4", render_mode="human")
    
    # Load the trained model
    print("Loading the 'humanoid_brain' model...")
    try:
        model = PPO.load("humanoid_brain")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have run 'train_humanoid.py' first.")
        env.close()
        return

    observation, info = env.reset()
    
    import time
    print("Running visualization...")
    # Run a visualization loop for longer so we can see it
    for step in range(10000):
        # Add a tiny delay so it doesn't run through 1000 steps in 0.1 seconds
        time.sleep(0.01)
        
        # Let the model predict the action based on the observation
        # Setting deterministic=True uses the mean of the policy's action distribution
        action, _states = model.predict(observation, deterministic=True)
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        # If the environment terminates or truncates, reset it
        if terminated or truncated:
            observation, info = env.reset()
            
    print("Finished visualization.")
    env.close()

if __name__ == "__main__":
    main()
