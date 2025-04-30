import numpy as np
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
import imageio
from fixed_controllers import *
import torch

# ---- SIMULATE BEST ROBOT ----


def simulate_best_robot(robot_structure, scenario=None, steps=500, controller=alternating_gait):

    connectivity = get_full_connectivity(robot_structure)
    # if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps,
                   body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0

    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        actuation = controller(action_size, t)

        ob, reward, terminated, truncated, info = env.step(actuation)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    # (max_height - initial_height) #-  abs(np.mean(positions[0, :])) # Max height gained is jump performance
    return t_reward


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500, controller=alternating_gait):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        # if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps,
                       body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        action_size = sim.get_dim_action_space(
            'robot')  # Get correct action size
        t_reward = 0

        frames = []
        for t in range(200):
            actuation = controller(action_size, t)
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


def get_param_as_weights(param_vector, model):
    shapes = [p.shape for p in model.parameters()]
    new_weights = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        new_weights.append(param_vector[idx:idx+size].reshape(shape))
        idx += size
    return new_weights


def set_weights(model, weights):
    """Update PyTorch model weights from a list of NumPy arrays."""
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            # Ensure the tensor being copied has the same dtype and device as the parameter
            param.copy_(torch.tensor(
                w, dtype=param.dtype, device=param.device))


def simulate_best_robot_nn(robot_structure, scenario, steps=500, controller=None):
    connectivity = get_full_connectivity(robot_structure)
    print(f"Scenario passed: {scenario}, type: {type(scenario)}")

    env = gym.make(scenario, max_episode_steps=steps,
                   body=robot_structure, connections=connectivity)
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
    try:
        connectivity = get_full_connectivity(robot_structure)

        # Convert flat weights array to the list format
        weights_list = get_param_as_weights(weights, brain)
        set_weights(brain, weights)

        env = gym.make(scenario, max_episode_steps=steps,
                       body=robot_structure, connections=connectivity)
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        state = env.reset()[0]
        frames = []

        t_reward = 0

        frames = []
        for t in range(400):
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)

            state_tensor = state_tensor.to(next(brain.parameters()).device)

            # Convert to tensor
            action = brain(state_tensor).detach(
            ).numpy().flatten()  # Get action

            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()

        if frames:
            imageio.mimsave(filename, frames, duration=duration, optimize=True)
        else:
            print(
                f"Warning: No frames generated for GIF '{filename}'. Simulation might have ended immediately.")

    except Exception as e:
        print(f'Error creating GIF: {e}')
        import traceback
        traceback.print_exc()


def set_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.tensor(w))


def create_gif_brain(robot_structure, brain, filename='best_robot.gif', duration=0.066, scenario=None, steps=500):
    try:
        """
        Create a smooth GIF of the robot simulation at 30fps.
        This function uses a NeuralController (brain) to control the robot's actions.
        """
        connectivity = get_full_connectivity(robot_structure)
        env = gym.make(scenario, max_episode_steps=steps,
                       body=robot_structure, connections=connectivity)
        state = env.reset()[0]  # Get initial state
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        t_reward = 0
        frames = []
        for t in range(steps):
            # Update actuation before stepping
            state_tensor = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = brain(state_tensor).detach(
            ).numpy().flatten()  # Get action
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward

            # Render and save frame
            frame = viewer.render('rgb_array')
            frames.append(frame)

            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid')
