import numpy as np
import torch
import yaml
import gymnasium as gym
import math
from gymnasium import spaces
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from JSCC.model import DeepJSCC
from JSCC.dataset import Vanilla
from JSCC.train import evaluate_epoch
from JSCC.utils import get_psnr

def calculate_propulsion_energy(P0, U_tip, d0, rho, s, G, P1, v0, v_m_u):
    """
    计算无人机推进能耗 \( E_{um}^{\text{r-uav}} \) 的函数

    参数说明：
    P0: 悬停时的叶片轮廓功率（Blade Profile Power）
    U_tip: 旋翼叶片的尖端速度（Tip Speed of Rotor Blade）
    d0: 机身阻力比（Fuselage Drag Ratio）
    rho: 空气密度（Air Density）
    s: 旋翼实度（Rotor Solidity）
    G: 旋翼盘面积（Rotor Disc Area）
    P1: 悬停时的感应功率（Induced Power）
    v0: 悬停时的平均旋翼感应速度（Mean Rotor Induced Velocity in Hover）
    v_m_u: 无人机在对应阶段的水平速度

    返回：
    计算得到的无人机推进能耗值
    """
    # 计算第一部分：P0*(1 + 3*(v_m_u)^2 / U_tip^2 )
    part1 = P0 * (1 + (3 * (v_m_u ** 2)) / (U_tip ** 2))

    # 计算第二部分：(1/2)*d0*rho*s*G*(v_m_u)^3
    part2 = 0.5 * d0 * rho * s * G * (v_m_u ** 3)

    # 计算第三部分：P1*(sqrt(1 + (v_m_u)^4/(4*v0^4)) - (v_m_u)^2/(2*v0^2))^(1/2)
    inner_sqrt = math.sqrt(1 + (v_m_u ** 4) / (4 * (v0 ** 4))) - (v_m_u ** 2) / (2 * (v0 ** 2))
    part3 = P1 * (inner_sqrt ** 0.5)

    # 总能耗为三部分之和
    total_energy = part1 + part2 + part3
    return total_energy



def calculate_path_loss(geometric_distance, eta_LoS, f_carrier):
    """
    计算空地通信的路径损耗（单位：dB）
    
    参数:
        geometric_distance (float): 无人机与用户之间的几何距离（m）
        eta_LoS (float): 视距链路的环境损耗（dB）
        f_carrier (float): 载波频率（Hz）
    
    返回:
        float: 路径损耗值（dB）
    """
    
    # 计算常数项 C = 20*log10(4πf_c / c)
    c = 3e8  # 光速（m/s）
    C = 20 * math.log10((4 * math.pi * f_carrier) / c)
    
    # 路径损耗公式
    G_i = 20 * math.log10(geometric_distance) + eta_LoS + C
    
    return G_i

def calculate_snr(eirp, G_i, B, sigma):
    """
    计算信噪比（SNR）

    参数:
        eirp (float): 发射功率（dBm）
        G_i (float): 天线增益（dB）
        B (float): 带宽（Hz）
        sigma (float): 噪声功率谱密度（dBm/Hz）

    返回:
        float: 信噪比（dB）
    """
    # eirp 输入为 dBm，需转换为瓦特
    eirp_watt = 10 ** ((eirp - 30) / 10)
    # G_i 输入为 dB，需转换为线性值
    G_i_linear = 10 ** (G_i / 10)
    sigma_watt = 10 ** ((sigma - 30) / 10)
    snr = 10 * math.log10((eirp_watt * G_i_linear) / (B * sigma_watt))
    return snr


config_path = r"/mnt/7t/tz/github/uav-jscc-fl/DDPG-Pytorch/JSCC/out/configs/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024.yaml"
times = 10
dataset_name = 'cifar10'
output_dir = '/mnt/7t/tz/github/uav-jscc-fl/DDPG-Pytorch/JSCC/out'
channel_type = 'AWGN'



class UAVEnvironmentGym(gym.Env):
    def __init__(self, params, num_users=20, max_time=160):
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
        self.ratio = params['ratio']  # 压缩比
        self.eta_loss = params['eta_loss']  # 环境损耗(dB)
        self.eirp = params['eirp']  # 发射功率(dBm)
        self.f_c = params['f_c']  # 载波频率(Hz)
        self.bandwidth = params['bandwidth']  # 带宽(Hz)
        self.noise_sigma = params['noise_sigma']  # 噪声功率谱密度(dBm/Hz)

        self.blade_power = params['blade_power']  # UAV桨叶功率消耗(W)
        self.induced_power = params['induced_power']  # UAV诱导功率消耗(W)
        self.tip_speed = params['tip_speed']  # UAV桨叶尖速(m/s)
        self.hover_speed = params['hover_speed']  # UAV悬停速度(m/s)
        self.drag_ratio = params['drag_ratio']  # UAV阻力比
        self.rotor_solidity = params['rotor_solidity']  # UAV桨叶实心度
        self.rotor_area = params['rotor_area']  # UAV桨叶面积(m^2)
        self.air_density = params['air_density']  # 空气密度(kg/m^3)

        self.max_energy = params['max_energy']  # 最大能量（J）
        self.energy = self.max_energy

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

        self.energy = self.max_energy

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
        cr = self.ratio  # 固定压缩比
        
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

        # 能耗
        proplusion_energy = calculate_propulsion_energy(
            P0=self.blade_power, 
            U_tip=self.tip_speed, 
            d0=self.drag_ratio, 
            rho=self.air_density, 
            s=self.rotor_solidity, 
            G=self.rotor_area, 
            P1=self.induced_power, 
            v0=self.hover_speed, 
            v_m_u=v
        )

        model_energy = 10 # 假设JSCC模型能耗为30J（可根据实际情况调整）
        step_energy = proplusion_energy + model_energy
        self.energy -= step_energy  # 扣除本步能耗

        # 通信质量计算
        active_users = [k for k in range(self.num_users) if self.effective_comm[k] < 1]

        pbar = tqdm(active_users, desc=f"Step {self.current_time:3d} PSNR Eval", 
            leave=False)
        
        for k in pbar:
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            if dist < 1e-6:
                dist = 1e-6
                         
            G_i = calculate_path_loss(dist, self.eta_loss, self.f_c)
            snr = calculate_snr(self.eirp, G_i, self.bandwidth, self.noise_sigma)
            
            # 更新进度条描述
            pbar.set_postfix({
                'User': k, 
                'SNR': f'{snr:.1f}dB', 
                'Dist': f'{dist:.1f}m'
            })
            
            # PSNR计算
            psnr = self._calculate_psnr_with_jscc(times=1, snr=snr)
            
            # 只考虑PSNR，不考虑delay
            if psnr >= self.eps_k[k]:
                self.effective_comm[k] += 1
                if self.effective_comm[k] >= 1:
                    self.completed_users += 1
                    pbar.set_postfix({
                        'User': k, 
                        'PSNR': f'{psnr:.1f}dB', 
                        'Status': '✓ Completed'
                    })
        
        # 判断是否所有用户已完成
        all_users_completed = (self.completed_users == self.num_users)
        # 判断无人机是否回到起点（允许一定误差）
        at_origin = np.linalg.norm(self.uav_state[:3] - np.array([0.0, 0.0, self.z_min])) < 5.0

        # 奖励计算
        lambda_energy = 0.001  # 能耗惩罚系数
        lambda_complete = 1.0  # 完成全部服务的奖励系数
        lambda_time = 0.05     # 时间效率奖励系数
        lambda_return = 1.0    # 回到起点奖励系数

        # 能耗惩罚
        r_energy = -lambda_energy * step_energy

        # 服务完成奖励
        r_complete = lambda_complete * int(all_users_completed and at_origin)

        # 时间效率奖励（只有全部完成且回到起点才奖励）
        r_time = lambda_time * (self.max_time - self.current_time) if (all_users_completed and at_origin) else 0

        # 回到起点奖励（可选）
        r_return = lambda_return * int(all_users_completed and at_origin)

        # 总奖励
        reward = r_energy + r_complete + r_time + r_return

        self.prev_completed = self.completed_users
        self.current_time += 1

        # 只有所有用户完成且回到起点才终止
        terminated = all_users_completed and at_origin
        truncated = (self.current_time >= self.max_time or self.energy <= 0)

        return self._get_state(), reward, terminated, truncated, {}