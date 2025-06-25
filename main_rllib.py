import ray
from ray import tune
from ray.tune.registry import register_env
from uav_gym import UAVEnvironmentGym

print(f"Ray 版本: {ray.__version__}")

def env_creator(env_config):
    params = env_config.get("params", {})
    num_users = env_config.get("num_users", 3)
    max_time = env_config.get("max_time", 100)
    return UAVEnvironmentGym(params, num_users=num_users, max_time=max_time)

register_env("UAV-v1", env_creator)

def main():
    ray.init(address="auto", ignore_reinit_error=True)
    
    print("集群资源:", ray.cluster_resources())
    cluster_resources = ray.cluster_resources()
    available_cpus = int(cluster_resources.get('CPU', 0))
    available_gpus = int(cluster_resources.get('GPU', 0))
    
    print(f"可用 CPU: {available_cpus}, 可用 GPU: {available_gpus}")
    
    # UAV 环境参数
    params = {
        'ratio': 0.8333333333333333,
        'eirp': 25,
        'f_c': 2.4e9,
        'bandwidth': 20e6,
        'noise_sigma': -169,
        'eta_loss': 1,
        'blade_power': 0.1,
        'induced_power': 0.1,
        'tip_speed': 100,
        'hover_speed': 10,
        'drag_ratio': 0.1,
        'rotor_solidity': 0.1,
        'rotor_area': 0.1,
        'air_density': 1.225,
        'max_energy': 10000,
        'max_horizontal_speed': 5,
        'max_vertical_speed': 3,
        'max_acceleration': 10,
        'time_slot': 1,
        'x_max': 100, 'x_min': 0,
        'y_max': 100, 'y_min': 0,
        'z_max': 50, 'z_min': 10
    }
    
    # 配置参数
    num_workers = 8
    num_envs_per_worker = 2
    rollout_fragment_length = 250  # 按照错误提示使用 250
    
    # 计算正确的训练批大小
    # train_batch_size = num_workers * num_envs_per_worker * rollout_fragment_length
    train_batch_size = num_workers * num_envs_per_worker * rollout_fragment_length
    print(f"计算得到的训练批大小: {train_batch_size}")
    
    config = {
        "env": "UAV-v1",
        "env_config": {
            "params": params,
            "num_users": 3,
            "max_time": 100
        },
        "framework": "torch",
        
        # 资源配置
        "num_gpus": 1 if available_gpus > 0 else 0,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "num_cpus_per_worker": 2,
        
        # Rollout 配置
        "rollout_fragment_length": rollout_fragment_length,  # 使用建议的值
        "batch_mode": "truncate_episodes",
        
        # PPO 超参数
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "entropy_coeff": 0.01,
        "vf_loss_coeff": 0.5,
        
        # 批处理设置 - 使用计算得到的值
        "train_batch_size": train_batch_size,  # 4000
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        
        # 网络配置
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "vf_share_layers": False,
        },
        
        # 评估配置
        "evaluation_interval": 20,
        "evaluation_duration": 10,
        "evaluation_num_workers": 2,
        "evaluation_config": {
            "explore": False,
        },
    }
    
    print("开始训练...")
    print(f"配置: {num_workers} workers, {num_envs_per_worker} envs/worker, rollout_length={rollout_fragment_length}")
    print(f"训练批大小: {train_batch_size}")
    
    # 启动训练
    results = tune.run(
        "PPO",
        name="UAV_PPO_fixed_batch",
        config=config,
        stop={
            "timesteps_total": 1000000,
            "training_iteration": 1000,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        verbose=2,
        fail_fast=False,
        max_failures=3,
    )
    
    print("训练完成！")
    best_trial = results.get_best_trial("episode_reward_mean", mode="max")
    if best_trial:
        print("最佳结果:", best_trial.last_result)

if __name__ == "__main__":
    main()