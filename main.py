from utils import str2bool, evaluate_policy
from datetime import datetime
from DDPG import DDPG_agent
import gymnasium as gym
import os, shutil
import argparse
import yaml
import torch
from uav_gym import UAVEnvironmentGym  # Import UAV environment if needed

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda:3', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=6, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3, UAV')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=50000, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=10000, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=2000, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=10000 , help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.15, help='exploring noise')
parser.add_argument('--max_times', type=int, default=1000, help='max time steps in one episode')
parser.add_argument('--users', type=int, default=3, help='number of users in UAV environment')
# parser.add_argument('--debug', type=str2bool, default=True, help='debug mode or not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device

params = {}
params['ratio'] = 0.8333333333333333 # 压缩比
params['eirp'] = 10 # 发射功率(dBm)
params['f_c'] = 2.4e9 # 载波频率(Hz)
params['bandwidth'] = 20e6 # 带宽(Hz)
params['noise_sigma'] = -169 # 噪声功率谱密度(dBm/Hz)
params['eta_loss'] = 1 # 传输损耗(dB)
params['blade_power'] = 60 # UAV桨叶功率消耗(W)
params['induced_power'] = 88 # UAV诱导功率消耗(W)
params['tip_speed'] = 100 # UAV桨叶尖速(m/s)
params['hover_speed'] = 10 # UAV悬停速度(m/s)
params['drag_ratio'] = 0.1 # UAV阻力比
params['rotor_solidity'] = 0.1 # UAV桨叶实心度
params['rotor_area'] = 0.1 # UAV桨叶面积(m^2)
params['air_density'] = 1.225 # 空气密度(kg/m^3)
params['max_energy'] = 358200 # 最大能量（J）
params['max_horizontal_speed'] = 5 # 最大水平速度（m/s）
params['max_vertical_speed'] = 3 # 最大垂直速度（m/s）
params['max_acceleration'] = 10 #  最大加速度（m/s^2）
params['time_slot'] = 1 # 时间片（s） # 测试的时候时间片可以设置大一点
params['x_max'] = 1000 # x轴最大值
params['x_min'] = 0   # x轴最小值
params['y_max']  = 1000# y轴最大值
params['y_min']  = 0  # y轴最小值
params['z_max']  = 100 # z轴最大值
params['z_min']  = 50 # z轴最小值

params['debug'] = True # 是否开启debug模式

def main():
    import numpy as np
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3','UAV-v1']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3', 'UAV']

    opt.loss_interval = 10 # 训练损失记录间隔
    # Build Env
    if opt.EnvIdex == 6:  # UAV环境
        env = UAVEnvironmentGym(params, num_users=opt.users, max_time=opt.max_times)
        eval_env = UAVEnvironmentGym(params, num_users=opt.users, max_time=opt.max_times)
    else:
        env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
        eval_env = gym.make(EnvName[opt.EnvIdex])
    
    # 状态空间维度
    opt.state_dim = env.observation_space.shape[0]
    # 动作空间维度
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    print(f'Env:{EnvName[opt.EnvIdex]}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{env._max_episode_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, turns=1)
            print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    else:
        total_steps = 0
        actor_loss, critic_loss = None, None  # 先初始化为 None
        episode_reward = 0  # 初始化累积奖励
        episode_steps = 0   # 初始化episode步数
        episode_count = 0   # 初始化episode计数

        best_reward = -np.inf
        best_trajectory = None
        
        # 外层循环，直到总训练步数达到最大训练步数
        while total_steps < opt.Max_train_steps:
            # 重置环境 获取初始状态和环境信息
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False
            episode_reward = 0  # 重置累积奖励
            episode_steps = 0   # 重置episode步数

            # === 轨迹记录 ===
            trajectory = []
            
            '''Interact & trian'''
            # 每步与环境交互，直到episode结束
            while not done:  
                if total_steps < opt.random_steps: 
                    a = env.action_space.sample() # 随机采样动作
                else: 
                    a = agent.select_action(s, deterministic=False) # 智能体选择动作
                s_next, r, dw, tr, info = env.step(a)
                done = (dw or tr)   
                agent.replay_buffer.add(s, a, r, s_next, dw)
                
                # 累积奖励和步数
                episode_reward += r
                episode_steps += 1
                s = s_next
                total_steps += 1

                # === 记录无人机三维轨迹 ===
                if hasattr(env, 'uav_state'):
                    trajectory.append(env.uav_state[:3].copy())

                '''train'''
                if total_steps >= opt.random_steps:
                    actor_loss, critic_loss = agent.train()
                    if opt.write and total_steps % opt.loss_interval == 0:
                        writer.add_scalar('train/actor_loss', actor_loss, global_step=total_steps)
                        writer.add_scalar('train/critic_loss', critic_loss, global_step=total_steps)
                        writer.add_scalar('train/step_reward', r, global_step=total_steps)

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write: 
                        writer.add_scalar('eval/episode_reward', ep_r, global_step=total_steps)
                    if actor_loss is not None and critic_loss is not None:
                        print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps)}, '
                              f'Episode Reward:{ep_r}, Actor Loss:{actor_loss:.4f}, Critic Loss:{critic_loss:.4f}')
                    else:
                        print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps)}, '
                              f'Episode Reward:{ep_r}, Actor Loss:-, Critic Loss:-')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps))
            
            # 一个episode结束后，记录累积奖励
            if opt.write:
                writer.add_scalar('train/episode_reward', episode_reward, global_step=total_steps)
                writer.add_scalar('train/episode_length', episode_steps, global_step=total_steps)
            episode_count += 1
            print(f'Episode {episode_count} finished after {episode_steps} steps, reward: {episode_reward:.2f}')

            # === 只保存最优轨迹 ===
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_trajectory = np.array(trajectory)
                np.save('best_trajectory.npy', best_trajectory)
                # 保存用户位置
                if hasattr(env, 'users'):
                    np.save('best_users.npy', env.users)
                print(f'Best trajectory updated at episode {episode_count}, reward: {best_reward:.2f}')

        env.close()
        eval_env.close()

if __name__ == '__main__':
    main()
    print('Training completed.')