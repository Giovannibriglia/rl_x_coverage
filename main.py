# Example env_configs: keys are setup names; each contains lists of (name, config) pairs
from src.simulation import Simulation

max_steps = 500
n_iters = 500
frames_per_batch = 6000

env_configs = {
    "basic": {
        "envs_train": [
            (
                "train_env",
                {
                    "max_steps": max_steps,
                    "n_agents": 3,
                    "seed": 42,
                    "frames_per_batch": frames_per_batch,
                    "scenario_name": "voronoi",
                    "env_kwargs": {
                        "n_gaussians": 1,
                        "grid_spacing": 0.05,
                        "centralized": False,
                        "shared_rew": False,
                        "n_obstacles": 0,
                        "n_iters": n_iters,
                        "n_rays": 50,
                        "lidar_range": 0.5,
                    },
                },
            )
        ],
        "envs_test": [
            (
                "train_env",
                {
                    "max_steps": max_steps,
                    "n_agents": 3,
                    "seed": 42,
                    "frames_per_batch": frames_per_batch,
                    "scenario_name": "voronoi",
                    "env_kwargs": {
                        "n_gaussians": 1,
                        "grid_spacing": 0.05,
                        "centralized": False,
                        "shared_rew": False,
                        "n_obstacles": 0,
                        "n_iters": n_iters,
                        "n_rays": 50,
                        "lidar_range": 0.5,
                    },
                },
            )
        ],
    }
}

# Example algo_configs: outer key is a grouping name; inner keys are agent identifiers with their configs
algo_configs = {
    "ippo": {
        "algo_name": "IPPO",
        "max_steps": max_steps,
        "n_agents": 3,
        "frames_per_batch": frames_per_batch,
        "n_iters": n_iters,
        "num_epochs": 50,
        "minibatch_size": 256,
        "max_grad_norm": 0.5,
        "clip_epsilon": 0.2,
        "entropy_eps": 0.01,
        "gamma": 0.99,
        "lambda": 0.95,
        "lr": 3e-4,
    }
}

# Example usage:
if __name__ == "__main__":
    sim = Simulation()
    sim.run(
        env_configs=env_configs,
        algo_configs=algo_configs,
        experiment_name="test_experiment",
    )
