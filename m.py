from src.simulation import Simulation
from src.utils import get_algo_dict, get_env_dict, get_torch_rl_env

if __name__ == "__main__":
    s = Simulation(experiment_name="try2", seed=0)
    device = s.device
    train_env_dict, train_env_name = get_env_dict("base", 3, 3, 250, 32)

    train_env = get_torch_rl_env(env_config=train_env_dict, device=device)

    algos_config = [
        get_algo_dict("./config/algos/mappo.yaml", train_env_dict, train_env),
        get_algo_dict("./config/algos/ippo.yaml", train_env_dict, train_env),
    ]

    test_env_dict1, test_env_name1 = get_env_dict("base", 3, 3, 1000, 8)
    test_env1 = get_torch_rl_env(env_config=test_env_dict1, device=device)
    test_env_dict2, test_env_name2 = get_env_dict("non_convex1", 5, 5, 1000, 8)
    test_env2 = get_torch_rl_env(env_config=test_env_dict2, device=device)

    test_envs = {
        test_env_name1: test_env1,
        test_env_name2: test_env2,
    }

    exp_dir = s.train_and_evaluate(algos_config, test_envs, n_checkpoints_train=25)
    s.plot_exp(exp_dir)
