import gymnasium as gym
from stable_baselines3 import PPO

def main():
    print("Initializing Humanoid-v4 environment...")
    # Initialize without render_mode for faster training
    env = gym.make("Humanoid-v4")
    
    # Create the PPO model (the "brain")
    # Multi-layer perceptron policy is good for this kind of continuous control
    print("Creating PPO Brain...")
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting training. This will take a while...")
    # Train for a moderate number of timesteps.
    # To get a somewhat walking humanoid, you might need 1,000,000+ steps.
    # We use 100,000 so it finishes reasonably fast but still shows *some* learning.
    model.learn(total_timesteps=100000)
    
    print("Training finished.")
    print("Saving the model...")
    model.save("humanoid_brain")
    print("Model saved as 'humanoid_brain.zip'.")
    
    env.close()

if __name__ == "__main__":
    main()
