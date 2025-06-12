from utils import str2bool,evaluate_policy
from datetime import datetime
from DDPG import DDPG_agent
import gymnasium as gym
import os, shutil
import argparse
import yaml
import torch
from uav_gym import UAVEnvironmentGym  # Import UAV environment if needed
from tensorboardX import SummaryWriter

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=6, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3, UAV')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=500, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e3, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=100, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

params = {}
params['ratio'] = 0.8333333333333333 # 压缩比
params['eirp'] = 25 # 发射功率(dBm)
params['f_c'] = 2.4e9 # 载波频率(Hz)
params['bandwidth'] = 20e6 # 带宽(Hz)
params['noise_sigma'] = -169 # 噪声功率谱密度(dBm/Hz)
params['eta_loss'] = 1 # 传输损耗(dB)

params['blade_power'] = 0.1 # UAV桨叶功率消耗(W)
params['induced_power'] = 0.1 # UAV诱导功率消耗(W)
params['tip_speed'] = 100 # UAV桨叶尖速(m/s)
params['hover_speed'] = 10 # UAV悬停速度(m/s)
params['drag_ratio'] = 0.1 # UAV阻力比
params['rotor_solidity'] = 0.1 # UAV桨叶实心度
params['rotor_area'] = 0.1 # UAV桨叶面积(m^2)

params['air_density'] = 1.225 # 空气密度(kg/m^3)
params['max_energy'] = 10000 # UAV最大能量(J)


def main():
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3','UAV-v1']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3', 'UAV']

    # Build Env
    if opt.EnvIdex == 6:  # UAV环境
        env = UAVEnvironmentGym(params, num_users=3, max_time=10)
        eval_env = UAVEnvironmentGym(params, num_users=3, max_time=10)
    else:
        env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
        eval_env = gym.make(EnvName[opt.EnvIdex])
        
    opt.state_dim = env.observation_space.shape[0]
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
        # 外层循环，直到总训练步数达到最大训练步数
        while total_steps < opt.Max_train_steps:
            # 重置环境 获取初始状态和环境信息
            s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            # 每步与环境交互，直到episode结束
            while not done:  
                if total_steps < opt.random_steps: 
                    a = env.action_space.sample() # 随机采样动作 先随机探索一段时间，再用智能体策略探索和利用
                else: 
                    a = agent.select_action(s, deterministic=False) # 智能体选择动作
                # 下一刻的状态 奖励 done或者win 是否因为达到最大步数截断 额外信息
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated

                # episode 终止条件：完成/失败 或 达到最大步数
                done = (dw or tr)   
                # 将当前状态，动作，奖励，下一时刻状态 根据是否结束的标签放入经验回放池
                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next
                total_steps += 1

                '''train'''
                if total_steps >= opt.random_steps:
                    actor_loss, critic_loss = agent.train()
                    
                    # 记录训练损失到 TensorBoard
                    if opt.write and total_steps % 100 == 0:  # 每100步记录一次损失
                        writer.add_scalar('train/actor_loss', actor_loss, global_step=total_steps)
                        writer.add_scalar('train/critic_loss', critic_loss, global_step=total_steps)
                        writer.add_scalar('train/total_reward', r, global_step=total_steps)  # 记录即时奖励

                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    ep_r = evaluate_policy(eval_env, agent, turns=3)
                    if opt.write: 
                        writer.add_scalar('eval/episode_reward', ep_r, global_step=total_steps)
                    print(f'EnvName:{BrifEnvName[opt.EnvIdex]}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}, Actor Loss:{actor_loss:.4f}, Critic Loss:{critic_loss:.4f}')

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000))
        env.close()
        eval_env.close()


if __name__ == '__main__':
    main()




