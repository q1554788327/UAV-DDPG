import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from collections import deque
import random

class UAVEnvironment:
    def __init__(self, num_users=20, max_time=160, device='cpu'):
        self.num_users = num_users
        self.max_time = max_time
        self.device = device
        
        # 论文参数映射
        self.x_max, self.y_max, self.z_max = 1000, 1000, 600  # 飞行空间限制
        self.z_min = 250  # 起飞/降落高度
        self.V_max = 50  # 最大速度 (m/s)
        self.a_max = 20  # 最大加速度 (m/s²)
        self.compression_ratios = [1/6, 7/48, 1/8, 5/48, 1/12, 1/16, 1/24, 1/48]  # 离散压缩比集合
        self.N = len(self.compression_ratios)  # 压缩比数量
        
        # 初始化用户位置和任务需求（随机生成，类似论文仿真设置）
        self.users = np.random.uniform(low=0, high=self.x_max, size=(num_users, 3)) # 初始化用户的 x, y 位置
        self.users[:, 2] = 0  # 用户高度统一为0
        # 服务质量需求
        self.eps_k = np.random.uniform(low=20, high=35, size=num_users)  # 最小PSNR需求
        self.T_k = np.random.uniform(low=10, high=50, size=num_users)  # 最大时延限制
        
        self.reset()
        
    def reset(self):
        # 无人机初始位置：(0, 0, z_min)
        self.uav_state = np.array([0.0, 0.0, self.z_min, 0.0, 0.0, 0.0])  # x, y, z, v, phi, theta
        self.effective_comm = np.zeros(self.num_users)  # 有效通信次数
        self.completed_users = 0
        self.current_time = 0
        return self._get_state()
    
    # 状态空间设计
    def _get_state(self):
        # 构造状态向量：无人机状态 + 用户状态 + 完成数 + 当前时间
        user_states = []
        for k in range(self.num_users):
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k]) # 计算无人机与用户的距离
            user_states.extend([self.users[k][0], self.users[k][1], dist, self.effective_comm[k]])
        state = np.concatenate([
            self.uav_state,  # 6维：x, y, z, v, phi, theta
            np.array(user_states),  # 4*K维：用户x, y, 距离, 有效通信次数
            np.array([self.completed_users, self.current_time])  # 2维
        ])
        return torch.tensor(state, dtype=torch.float32, device=self.device)
    
    def step(self, action):
        # 解析动作：[a_U, d_phi, d_theta, cr_idx]（假设cr_idx为离散索引）
        a_U, d_phi, d_theta, cr_idx = action
        cr_idx = int(cr_idx)  # 确保索引为整数
        cr = self.compression_ratios[cr_idx]  # 离散压缩比
        
        # 更新无人机状态（基于论文式(19)(24)(32)(33)）
        phi = self.uav_state[4] + d_phi * 2 * np.pi  # 航向角增量映射
        theta = self.uav_state[5] + d_theta * np.pi/2  # 俯仰角增量映射
        a = a_U * self.a_max  # 实际加速度
        v = self.uav_state[3] + a * 1  # 假设时间步长为1秒
        v = np.clip(v, 0, self.V_max)
        dx = v * np.cos(phi) * np.cos(theta)
        dy = v * np.sin(phi) * np.cos(theta)
        dz = v * np.sin(theta)
        x = self.uav_state[0] + dx
        y = self.uav_state[1] + dy
        z = self.uav_state[2] + dz
        z = np.clip(z, self.z_min, self.z_max)
        self.uav_state = np.array([x, y, z, v, phi, theta])
        
        # 计算通信质量（简化版，基于论文式(2)(3)(5)-(15)）
        rewards = 0
        for k in range(self.num_users):
            if self.effective_comm[k] >= 1:  # 任务已完成
                continue
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            # 简化LoS概率和路径损耗计算
            los_prob = 1 / (1 + 24.81 * np.exp(-0.11 * (np.arctan2(self.uav_state[2], dist) * 180/np.pi - 24.81)))
            path_loss = 20 * np.log10(4 * np.pi * 4.9e9 * dist / 3e8) + (los_prob * 1 + (1-los_prob)*20)
            recv_power = 20 - path_loss  # 发射功率20dBm
            snr = recv_power - (-100)  # 噪声功率-100dBm
            psnr = 10 * np.log10(255**2 / (0.1 + 0.5/(snr+1)))  # 简化PSNR模型
            delay = (3*32*32 * cr) / 3 * 97.6e-6  # 基于式(3)，S0=3, T0=97.6us
            if psnr >= self.eps_k[k] and delay <= self.T_k[k]:
                self.effective_comm[k] += 1
                if self.effective_comm[k] >= 1:
                    self.completed_users += 1
            
        # 计算奖励（基于论文式(35)-(40)）
        lambda1, lambda2, lambda3, lambda4, lambda5 = 0.06, 0.0012, 0.045, 0.0005, 0.025
        r1 = lambda1 * (self.max_time - self.current_time) if self.completed_users == self.num_users else 0
        r2 = lambda2 * (self.completed_users - self.prev_completed) * (self.max_time - self.current_time)
        r3 = lambda3 * (self.max_time - self.current_time) if self.completed_users == self.num_users else 0
        dist_start = np.linalg.norm(self.uav_state[:2])  # x,y距离起点
        r4 = -lambda4 * dist_start
        r5 = lambda5 * np.sum(1 - (self.effective_comm >= 1).astype(float))
        reward = r1 + r2 + r3 + r4 + r5
        
        self.prev_completed = self.completed_users
        self.current_time += 1
        done = (self.current_time >= self.max_time) or (self.completed_users == self.num_users)
        return self._get_state(), reward, done, {}
    
class DDPGActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 输出连续动作：[a, d_phi, d_theta, cr_cont]
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 归一化到[-1, 1]
        )
        
    def forward(self, state):
        x = self.layers(state)
        return self.mu(x)  # 输出连续动作参数

class DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGCritic, self).__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        s = self.state_net(state)
        a = self.action_net(action)
        return self.out(torch.cat([s, a], dim=-1))
    
class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=1e-4, gamma=0.99, tau=0.005, sigma=0.2, device='cpu'):
        self.device = device
        self.actor = DDPGActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = DDPGCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # 初始化目标网络参数与主网络相同
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma  # 探索噪声标准差
        self.memory = deque(maxlen=100000)
        
    def select_action(self, state, explore=True):
        state = state.unsqueeze(0).to(self.device)
        action = self.actor(state).squeeze().detach().cpu().numpy()  # 添加.detach()
        if explore:
            action += np.random.normal(0, self.sigma, size=action.shape)  # 添加高斯噪声
        # 压缩比参数映射：将cr_cont∈[-1,1]映射到[0,1]，再转换为索引
        cr_cont = (action[-1] + 1) / 2  # 从[-1,1]到[0,1]
        cr_idx = np.clip(int(np.round(cr_cont * (env.N-1))), 0, env.N-1)
        action = np.concatenate([action[:-1], [cr_cont]])  # 保留连续参数用于网络训练
        return np.array([action[0], action[1], action[2], cr_idx])  # 返回实际执行的动作（含离散索引）
    
    def store_transition(self, state, action, reward, next_state, done):
        # 存储连续动作参数（cr_cont）而非离散索引
        cr_cont = (action[3] / (env.N-1))  # 离散索引转连续参数（反归一化）
        cont_action = np.concatenate([action[:3], [cr_cont]])
        self.memory.append((state, cont_action, reward, next_state, done))
        
    def update(self, batch_size=1024):
        if len(self.memory) < batch_size: return
        batch = np.random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).view(-1, 1)
        
        # Critic更新：目标Q值计算
        next_actions = self.actor_target(next_states)
        q_next = self.critic_target(next_states, next_actions).detach()
        q_target = rewards + self.gamma * q_next * (1 - dones)
        q_current = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_current, q_target)
        
        # Actor更新：最大化Q值
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # 梯度下降
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络（参考目标网络机制）
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
# 初始化环境和算法（参考仿真参数）
env = UAVEnvironment(num_users=20, max_time=160, device=device)
state_dim = 6 + 4*20 + 2  # 状态维度（式29）
action_dim = 4  # [a, d_phi, d_theta, cr_cont]
ddpg = DDPG(state_dim, action_dim, hidden_dim=256, lr=1e-4, device=device)



# 训练参数（参考训练设置）
episodes = 20000
batch_size = 1024
exploration_decay = 0.995
min_sigma = 0.01

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    env.prev_completed = 0  # 重置上一时刻完成数
    while True:
        # 选择动作并执行
        action = ddpg.select_action(state, explore=(ddpg.sigma > min_sigma))
        next_state, reward, done, _ = env.step(action)
        ddpg.store_transition(state, action, reward, next_state, done)
        
        # 更新算法
        ddpg.update(batch_size)
        if ddpg.sigma > min_sigma:
            ddpg.sigma *= exploration_decay
        
        total_reward += reward
        state = next_state
        if done:
            print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Sigma: {ddpg.sigma:.3f}")
            break