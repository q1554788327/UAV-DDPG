import numpy as np
import torch
import yaml
import gymnasium as gym
from gymnasium import spaces
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import yaml
from tensorboardX import SummaryWriter
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from JSCC.model import DeepJSCC
from JSCC.dataset import Vanilla
from JSCC.train import evaluate_epoch
from JSCC.utils import get_psnr

def eval_snr(model, test_loader, writer, param, times=10):
    snr_list = range(0, 26, 1)
    for snr in snr_list:
        model.change_channel(param['channel'], snr)
        test_loss = 0
        for i in range(times):
            test_loss += evaluate_epoch(model, param, test_loader)

        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
        writer.add_scalar('psnr', psnr, snr)

config_path = r"/mnt/7t/tz/github/uav-jscc-fl/DDPG-Pytorch/JSCC/out/configs/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024.yaml"
times = 10
dataset_name = 'cifar10'
output_dir = '/mnt/7t/tz/github/uav-jscc-fl/DDPG-Pytorch/JSCC/out'
channel_type = 'AWGN'



class UAVEnvironmentGym(gym.Env):
    def __init__(self, num_users=20, max_time=160):
        super(UAVEnvironmentGym, self).__init__()
        
        self.num_users = num_users
        self.max_time = max_time

        # 每局游戏的最大步数
        self._max_episode_steps = max_time
        
        # 论文参数映射
        self.x_max, self.y_max, self.z_max = 1000, 1000, 300
        self.z_min = 250
        self.V_max = 50
        self.a_max = 20
        self.compression_ratio = 1 / 4  # 压缩比
        
        # 初始化JSCC模型（参考eval.py）
        self._process_config_for_init(config_path, output_dir, dataset_name)
        
        # 定义观测空间和动作空间
        # 状态维度：6(UAV) + 4*num_users + 2 = 88维
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(6 + 4*num_users + 2,), 
            dtype=np.float32
        )
        
        # 动作空间：连续动作 [a_U, d_phi, d_theta, cr_cont] ∈ [-1,1]^4
        # 动作空间包括：无人机的加速度 航向角变化 俯仰角变化
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # 初始化用户位置和需求
        self.users = np.random.uniform(low=0, high=self.x_max, size=(num_users, 3))
        self.users[:, 2] = 0
        self.eps_k = np.random.uniform(low=20, high=35, size=num_users)
        self.T_k = np.random.uniform(low=10, high=50, size=num_users)
        self.min_comm = np.ones(num_users)  # 修改为数组，每个用户的最小通信次数为1
        
        self.reset()
        
    def _process_config_for_init(self, config_path, output_dir, dataset_name):
        """修改后的process_config，只用于初始化，不执行评估"""
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            assert dataset_name == config['dataset_name']
            self.params = config['params']
            c = config['inner_channel']

        # 准备测试数据
        if dataset_name == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor()])
            self.test_dataset = datasets.CIFAR10(
                root='/mnt/7t/tz/github/uav-jscc-fl/dataset', 
                train=False,
                download=False,  # 假设已下载
                transform=transform
            )
            self.test_loader = DataLoader(
                self.test_dataset, 
                shuffle=True,
                batch_size=min(self.params['batch_size'], 8),  # 限制batch size提高速度
                num_workers=0  # 避免多进程问题
            )

        elif dataset_name == 'imagenet':
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((128, 128))
            ])
            self.test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
            self.test_loader = DataLoader(
                self.test_dataset, 
                shuffle=True,
                batch_size=min(self.params['batch_size'], 8),
                num_workers=0
            )
        else:
            raise Exception('Unknown dataset')

        # 创建和加载模型
        name = os.path.splitext(os.path.basename(config_path))[0]
        self.model = DeepJSCC(c=c)
        self.model = self.model.to(self.params['device'])
        
        # 加载权重
        pkl_list = glob.glob(os.path.join(output_dir, 'checkpoints', name, '*.pkl'))
        if pkl_list:
            self.model.load_state_dict(torch.load(pkl_list[-1], map_location=self.params['device']))
            self.model.eval()
            print(f"Model loaded from: {pkl_list[-1]}")
        else:
            raise Exception(f"No model found in: {os.path.join(output_dir, 'checkpoints', name)}")
        
            
    def _calculate_psnr_with_jscc(self, times=10, snr=10):
        """使用JSCC模型计算PSNR"""
        self.model.change_channel(self.params['channel'], snr)
        test_loss = 0
        for i in range(times):
            test_loss += evaluate_epoch(self.model, self.params, self.test_loader)
        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
            
        return psnr

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 无人机初始状态
        self.uav_state = np.array([0.0, 0.0, self.z_min, 0.0, 0.0, 0.0])
        self.effective_comm = np.zeros(self.num_users)
        self.completed_users = 0
        self.current_time = 0
        self.prev_completed = 0

        info = {
            "uav_state": self.uav_state.copy(),
            "effective_comm": self.effective_comm.copy(),
            "completed_users": self.completed_users,
            "current_time": self.current_time,
            "prev_completed": self.prev_completed
        }
        
        return self._get_state(), info
    
    def _get_state(self):
        user_states = []
        for k in range(self.num_users):
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            user_states.extend([self.users[k][0], self.users[k][1], dist, self.effective_comm[k]])
        
        state = np.concatenate([
            self.uav_state,
            np.array(user_states),
            np.array([self.completed_users, self.current_time])
        ])
        return state.astype(np.float32)
    
    def step(self, action):
        # 连续动作到混合动作的映射
        a_U, d_phi, d_theta, cr_cont = action
        
        # 压缩比定死
        cr = 1 / 4  # 固定压缩比
        
        # 更新无人机状态
        phi = self.uav_state[4] + d_phi * 2 * np.pi
        theta = self.uav_state[5] + d_theta * np.pi/2
        a = a_U * self.a_max
        v = np.clip(self.uav_state[3] + a * 1, 0, self.V_max)
        
        dx = v * np.cos(phi) * np.cos(theta)
        dy = v * np.sin(phi) * np.cos(theta)
        dz = v * np.sin(theta)
        
        x = self.uav_state[0] + dx
        y = self.uav_state[1] + dy
        z = np.clip(self.uav_state[2] + dz, self.z_min, self.z_max)
        
        self.uav_state = np.array([x, y, z, v, phi, theta])

        # 通信质量计算
        active_users = [k for k in range(self.num_users) if self.effective_comm[k] < 1]

        pbar = tqdm(active_users, desc=f"Step {self.current_time:3d} PSNR Eval", 
            leave=False)
        
        for k in pbar:
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            if dist < 1e-6:
                dist = 1e-6
                
            # LoS概率和路径损耗
            elevation_angle = np.arctan2(self.uav_state[2], dist) * 180/np.pi
            # 建立Los概率模型
            los_prob = 1 / (1 + 24.81 * np.exp(-0.11 * (elevation_angle - 24.81)))
            path_loss = 20 * np.log10(4 * np.pi * 4.9e9 * dist / 3e8) + (los_prob * 1 + (1-los_prob)*20)
            
            recv_power = 20 - path_loss
            snr = recv_power - (-100)
            
            # 更新进度条描述
            pbar.set_postfix({
                'User': k, 
                'SNR': f'{snr:.1f}dB', 
                'Dist': f'{dist:.1f}m'
            })
            
            # PSNR计算
            psnr = self._calculate_psnr_with_jscc(snr=snr)
            
            delay = (3*32*32 * cr) / 3 * 97.6e-6
            
            if psnr >= self.eps_k[k] and delay <= self.T_k[k]:
                self.effective_comm[k] += 1
                if self.effective_comm[k] >= 1:
                    self.completed_users += 1
                    pbar.set_postfix({
                        'User': k, 
                        'PSNR': f'{psnr:.1f}dB', 
                        'Status': '✓ Completed'
                    })
        
        # 奖励计算
        lambda1, lambda2, lambda3, lambda4, lambda5 = 0.06, 0.0012, 0.045, 0.0005, 0.025
        
        # 完成任务时间奖励，要求无人机返回起点
        dist_to_start = np.linalg.norm(self.uav_state[:3] - np.array([0, 0, self.z_min]))  # 计算到起点的距离
        return_threshold = 5  # 返回起点的距离阈值（可调整）
        
        if self.completed_users == self.num_users:
            if dist_to_start <= return_threshold:
                # 无人机已返回起点，给予完整的时间奖励
                r1 = lambda1 * (self.max_time - self.current_time)
        else:
            r1 = 0

        r2 = lambda2 * (self.completed_users - self.prev_completed) * (self.max_time - self.current_time) # 用户任务完成中间奖励
        r3 = lambda3 * (self.max_time - self.current_time) if self.completed_users == self.num_users else 0 # 全局任务完成奖励
        
        r4 = -lambda4 * dist_to_start # 起飞点回归奖励，鼓励无人机每次执行完任务都尽可能靠近起飞点
        r5 = -lambda5 * np.sum(self.min_comm - (self.effective_comm >= self.min_comm).astype(float))
        
        reward = r1 + r2 + r3 + r4 + r5
        
        self.prev_completed = self.completed_users
        self.current_time += 1
        
        terminated = (self.completed_users == self.num_users)
        truncated = (self.current_time >= self.max_time)
        
        return self._get_state(), reward, terminated, truncated, {}