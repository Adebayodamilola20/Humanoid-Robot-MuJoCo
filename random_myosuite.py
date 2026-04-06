import gymnasium as gym
import myosuite # Important to import myosuite so it registers the environments in Gym
import time

def main():
    print("Initializing MyoSuite musculoskeletal environment...")
    
    # myoLegWalk-v0 or myoBodyWalk-v1 use the detailed anatomical skeleton
    # If render_mode="human" gives you an error, you can just delete that parameter and uncomment env.render()
    env = gym.make("myoLegWalk-v0", render_mode="human")
    
    observation, info = env.reset()
    
    for step in range(1000):
        # Sample a random action from the environment's action space
        action = env.action_space.sample()
        
        # Take the action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # env.render() # Uncomment this if the window doesn't appear
        
        # Add a delay so human eyes can see the physics before it exits
        time.sleep(0.01)
        
        if terminated or truncated:
            observation, info = env.reset()
            
    env.close()

if __name__ == "__main__":
    main()
