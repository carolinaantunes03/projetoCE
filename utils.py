import numpy as np
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *
import torch 

# ---- SIMULATE BEST ROBOT ----
def simulate_best_robot(robot_structure, scenario=None, steps=500, controller = alternating_gait):
    
    connectivity = get_full_connectivity(robot_structure)
    #if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0
    
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size,t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward #(max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance



def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        #if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0

        frames = []
        for t in range(200):
            actuation = controller(action_size,t)
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')
        

def simulate_best_robot_nn(robot_structure, scenario, steps=500, controller=None):
        connectivity = get_full_connectivity(robot_structure)
        print(f"Scenario passed: {scenario}, type: {type(scenario)}")

        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        state = env.reset()[0]
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        total_reward = 0
        for t in range(steps):
            action = controller(state)  # NN controller expects observation
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                env.reset()
                break
            viewer.render('screen')

        viewer.close()
        env.close()

        return total_reward
    


def create_gif_nn(weights, brain, robot_structure, scenario, steps, filename='best_robot.gif', duration=0.066):
    connectivity = get_full_connectivity(robot_structure)  # You forgot this before too
    set_weights(brain, weights)  # Load weights into the network

    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    state = env.reset()[0]
    frames = []

    for t in range(steps):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = brain(state_tensor).detach().numpy().flatten()
        frame = viewer.render('rgb_array')  # Capture frame
        frames.append(frame)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    viewer.close()
    env.close()

    imageio.mimsave(filename, frames, duration=duration, optimize=True)
    
def set_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.tensor(w))


