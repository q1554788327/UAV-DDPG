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
from JSCC.eval import evaluate_epoch
from JSCC.utils import get_psnr

# 获取当前文件所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接配置文件的绝对路径
config_path = os.path.join(base_dir, "configs/config.yaml")
model_path = os.path.join(base_dir, "JSCC/out/checkpoints/model.pkl")

# 编写用于测试的计算psnr函数
def test_caculate_psnr(snr):
    # 改进的PSNR计算，基于SNR的合理模型
    base_psnr = 10.0  # 基础PSNR
    snr_contribution = max(0, snr) * 0.8  # SNR贡献
    max_achievable_psnr = 40.0  # 上限
    
    psnr = base_psnr + snr_contribution
    psnr = min(psnr, max_achievable_psnr)
    psnr = max(psnr, 5.0)  # 下限，确保PSNR不会太低
    
    return psnr

def calculate_propulsion_energy(P0, U_tip, d0, rho, s, G, P1, v0, v_m_u):
    """
    计算无人机推进能耗 $ E_{um}^{\text{r-uav}} $ 的函数
    """
    part1 = P0 * (1 + (3 * (v_m_u ** 2)) / (U_tip ** 2))
    part2 = 0.5 * d0 * rho * s * G * (v_m_u ** 3)
    inner_sqrt = math.sqrt(1 + (v_m_u ** 4) / (4 * v0 ** 4)) - (v_m_u ** 2) / (2 * v0 ** 2)
    part3 = P1 * (inner_sqrt ** 0.5)
    total_energy = part1 + part2 + part3
    return total_energy

def calculate_path_loss(geometric_distance: float, eta_LoS: float, f_carrier: float) -> float:
    """计算空地通信的路径损耗（单位：dB）"""
    c = 3e8
    C = 20 * math.log10((4 * math.pi * f_carrier) / c)
    G_i = 20 * math.log10(geometric_distance) + C + eta_LoS
    return G_i

def calculate_snr(eirp, G_i, B, sigma):
    """计算信噪比（SNR）"""
    P_watt = 10 ** ((eirp - 30) / 10)
    sigma_watt_per_hz = 10 ** ((sigma - 30) / 10)
    path_loss = 10 ** (-G_i / 10)
    total_noise_power = sigma_watt_per_hz * B
    signal_power = P_watt * path_loss
    snr = 10 * math.log10(signal_power / total_noise_power)
    return snr

times = 1
dataset_name = 'cifar10'
channel_type = 'AWGN'

class UAVEnvironmentGym(gym.Env):
    def __init__(self, params, num_users=20, max_time=160):
        super(UAVEnvironmentGym, self).__init__()
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.batch_size = config['batch_size']
            self.c = config['inner_channel']
            self.dataset = config['dataset']
            self.channel_type = config['channel_type']
        
        self.num_users = num_users
        self.max_time = max_time
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_episode_steps = max_time
        
        self.x_max = params['x_max']
        self.x_min = params['x_min']
        self.y_max = params['y_max']
        self.y_min = params['y_min']
        self.z_max = params['z_max']
        self.z_min = params['z_min']

        if params['debug']:
            # 简化地图尺寸
            self.x_max = 50
            self.x_min = 0
            self.y_max = 50
            self.y_min = 0
            self.z_max = 30
            self.z_min = 10

        self.ratio = params['ratio']
        self.eta_loss = params['eta_loss']
        self.eirp = params['eirp']
        self.f_c = params['f_c']
        self.bandwidth = params['bandwidth']
        self.noise_sigma = params['noise_sigma']

        self.blade_power = params['blade_power']
        self.induced_power = params['induced_power']
        self.tip_speed = params['tip_speed']
        self.hover_speed = params['hover_speed']
        self.drag_ratio = params['drag_ratio']
        self.rotor_solidity = params['rotor_solidity']
        self.rotor_area = params['rotor_area']
        self.air_density = params['air_density']

        self.max_energy = params['max_energy']
        self.max_horizontal_speed = params['max_horizontal_speed']
        self.max_vertical_speed = params['max_vertical_speed']
        self.max_acceleration = params['max_acceleration']
        self.time_slot = params['time_slot']

        # 新增无人机姿态参数
        self.max_yaw_rate = np.pi/4  # 最大偏航角变化率(rad/s)
        self.max_pitch_rate = np.pi/8  # 最大俯仰角变化率(rad/s)
        self.roll = 0  # 横滚角(假设为0，简化模型)

        # =============== 周期性运动参数 ===============
        self.cycle_length = params.get('cycle_length', 80)  # 周期长度（步数）
        self.return_threshold = params.get('return_threshold', 3.0)  # 返回起点的距离阈值
        self.exploration_ratio = params.get('exploration_ratio', 0.75)  # 探索阶段占周期的比例
        
        print(f"Cycle parameters:")
        print(f"  Cycle length: {self.cycle_length} steps")
        print(f"  Return threshold: {self.return_threshold}m")
        print(f"  Exploration ratio: {self.exploration_ratio}")

        self._process_config_for_init(config_path, dataset_name)

        # 理论下限路径损耗
        self.G_i_min = calculate_path_loss(self.z_min, self.eta_loss, self.f_c)
        # 理论信噪比上限
        self.min_snr = calculate_snr(self.eirp, self.G_i_min, self.bandwidth, self.noise_sigma)
        # 理论 PSNR 上限
        self.max_psnr = test_caculate_psnr(snr=self.min_snr)
        print(f"Max PSNR: {self.max_psnr:.2f} dB")

        self.energy = self.max_energy

        # 用户位置初始化
        if params['debug']:
            # Debug模式：手动设置3个用户
            self.users = np.array([
                [25, 25, 0],  # 用户1：需要飞到地图中心
                [35, 15, 0],  # 用户2：右下角
                [35, 35, 0]   # 用户3：右上角，最远
            ])
            self.num_users = len(self.users)
        else:
            # 正常模式：随机分布用户
            x_coords = np.random.uniform(low=self.x_min, high=self.x_max, size=self.num_users)
            y_coords = np.random.uniform(low=self.y_min, high=self.y_max, size=self.num_users)
            z_coords = np.zeros(self.num_users)
            self.users = np.column_stack([x_coords, y_coords, z_coords])

        print(f"Number of users: {self.num_users}")
        print(f"User positions: {self.users}")

        # 修正：将起点位置改为不在地图边界上
        self.start_position = np.array([10.0, 10.0, self.z_min])  # 起点位置从(20,20)改为(10,10)
        
        # =============== 周期性运动状态管理 ===============
        self.current_cycle = 0  # 当前周期数
        self.cycle_step = 0     # 当前周期内的步数
        self.returning_to_start = False  # 是否正在返回起点
        self.at_start_position = True    # 是否在起点位置
        
        # 周期性PSNR记录 - 关键数据结构
        self.cycle_user_psnr_sum = np.zeros(self.num_users)     # 当前周期内每个用户的PSNR累积
        self.cycle_user_max_psnr = np.zeros(self.num_users)     # 当前周期内每个用户的最大PSNR
        self.cycle_user_min_distance = np.full(self.num_users, float('inf'))  # 周期内与每个用户的最小距离
        self.cycle_user_service_time = np.zeros(self.num_users) # 周期内每个用户的服务时间
        
        # 全局PSNR记录
        self.total_user_psnr_sum = np.zeros(self.num_users)     # 总体每个用户的PSNR累积
        self.total_user_max_psnr = np.zeros(self.num_users)     # 总体每个用户的最大PSNR
        
        # 周期历史记录
        self.cycle_history = []  # 记录每个周期的性能
        
        print(f"Mission parameters:")
        print(f"  Start position: {self.start_position}")
        print(f"  Mode: Periodic return to start every {self.cycle_length} steps")

        # 修改观察空间：包含位置、姿态、速度和周期信息
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(7 + 7*self.num_users + 10,),  # 7(UAV状态) + 7*用户数 + 10(全局+周期信息)
            dtype=np.float32
        )
        
        # 修改动作空间：偏航角、俯仰角和速度
        self.action_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi/4, 0]),  # [yaw, pitch, speed]
            high=np.array([np.pi, np.pi/4, self.max_horizontal_speed]),
            dtype=np.float32
        )

        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Action space: [yaw, pitch, speed]")
        
        self.reset()
    
    def _process_config_for_init(self, config_path, dataset_name):
        if dataset_name == 'cifar10':
            dataset_path = os.path.join(base_dir, "../datasets/cifar10")
            transform = transforms.Compose([transforms.ToTensor()])
            self.test_dataset = datasets.CIFAR10(
                root=dataset_path, 
                train=False,
                download=True,
                transform=transform
            )
            self.test_loader = DataLoader(
                self.test_dataset, 
                shuffle=True,
                batch_size=min(self.batch_size, 8),
                num_workers=0
            )
        elif dataset_name == 'imagenet':
            dataset_path = os.path.join(base_dir, "../datasets/imagenet")
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize((128, 128))
            ])
            self.test_dataset = Vanilla(root=dataset_path, transform=transform)
            self.test_loader = DataLoader(
                self.test_dataset, 
                shuffle=True,
                batch_size=min(self.batch_size, 8),
                num_workers=0
            )
        else:
            raise Exception('Unknown dataset')

        name = os.path.splitext(os.path.basename(config_path))[0]
        self.model = DeepJSCC(c=self.c)
        self.model = self.model.to(self.device)

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded from: {model_path}")
        else:
            raise Exception(f"No model found at: {model_path}")
            
    def _calculate_psnr_with_jscc(self, times=1, snr=10):
        self.model.change_channel(self.channel_type, snr)
        test_loss = 0
        for i in range(times):
            test_loss += evaluate_epoch(self.model, self.device, self.test_loader)
        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
        return psnr

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 修正：初始化UAV状态在起点附近，增加随机性
        init_x = self.start_position[0] + np.random.uniform(-2.0, 2.0)
        init_y = self.start_position[1] + np.random.uniform(-2.0, 2.0)
        init_yaw = np.random.uniform(-np.pi/6, np.pi/6)  # 随机初始朝向
        
        # 初始化UAV状态：[x, y, z, yaw, pitch, speed, roll]
        self.uav_state = np.array([init_x, init_y, self.z_min, init_yaw, 0.0, 0.0, 0.0])
        self.current_time = 0
        self.energy = self.max_energy

        # 重置周期性运动状态
        self.current_cycle = 0
        self.cycle_step = 0
        self.returning_to_start = False
        
        # 修正：正确设置初始at_start_position状态
        distance_to_start = np.linalg.norm(self.uav_state[:3] - self.start_position)
        self.at_start_position = (distance_to_start <= self.return_threshold)
        
        # 重置PSNR记录
        self.cycle_user_psnr_sum = np.zeros(self.num_users)
        self.cycle_user_max_psnr = np.zeros(self.num_users)
        self.cycle_user_min_distance = np.full(self.num_users, float('inf'))
        self.cycle_user_service_time = np.zeros(self.num_users)
        
        self.total_user_psnr_sum = np.zeros(self.num_users)
        self.total_user_max_psnr = np.zeros(self.num_users)
        
        self.cycle_history = []
        
        # 初始化历史变量
        self.prev_position = self.uav_state[:3].copy()
        self.prev_distance_to_start = distance_to_start
        self.prev_user_psnrs = np.zeros(self.num_users)

        print(f"Reset - UAV position: {self.uav_state[:3]}, Start position: {self.start_position}")
        print(f"Initial distance to start: {distance_to_start:.2f}m, at_start_position: {self.at_start_position}")

        info = {
            "uav_state": self.uav_state.copy(),
            "current_time": self.current_time,
            "current_cycle": self.current_cycle,
            "cycle_step": self.cycle_step,
            "energy": self.energy,
            "returning_to_start": self.returning_to_start,
            "at_start_position": self.at_start_position,
        }
        return self._get_state(), info
    
    def _get_state(self):
        user_states = []
        user_psnrs = []
        
        # 计算每个用户的状态
        for k in range(self.num_users):
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            if dist < 1e-6:
                dist = 1e-6
            
            # 计算PSNR
            G_i = calculate_path_loss(dist, self.eta_loss, self.f_c)
            snr = calculate_snr(self.eirp, G_i, self.bandwidth, self.noise_sigma)
            psnr = test_caculate_psnr(snr=snr)
            user_psnrs.append(psnr)
            
            # 计算周期内平均PSNR和服务率
            cycle_avg_psnr = self.cycle_user_psnr_sum[k] / max(self.cycle_step, 1)
            service_ratio = self.cycle_user_service_time[k] / max(self.cycle_step, 1)
            
            # 用户状态：x, y, 距离, 当前PSNR, 周期平均PSNR, 周期最大PSNR, 服务比例
            user_states.extend([
                self.users[k][0], 
                self.users[k][1], 
                dist, 
                psnr,
                cycle_avg_psnr,
                self.cycle_user_max_psnr[k],
                service_ratio
            ])

        # 计算PSNR统计信息
        min_psnr = min(user_psnrs)
        avg_psnr = np.mean(user_psnrs)
        
        # 计算到起点的距离
        distance_to_start = np.linalg.norm(self.uav_state[:3] - self.start_position)
        
        # 计算周期进度
        cycle_progress = self.cycle_step / self.cycle_length
        exploration_phase = self.cycle_step < self.cycle_length * self.exploration_ratio
        
        # 计算周期内总体PSNR性能
        cycle_total_avg_psnr = np.mean(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
        cycle_total_max_psnr = np.mean(self.cycle_user_max_psnr)
        
        # 计算PSNR不平衡度（标准差）
        cycle_psnr_std = np.std(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
        
        # 状态向量
        state = np.concatenate([
            # UAV状态：x, y, z, yaw, pitch, speed, roll
            self.uav_state,  
            # 用户状态：每个用户的x, y, 距离, 当前PSNR, 周期平均PSNR, 周期最大PSNR, 服务比例
            np.array(user_states),
            # 全局+周期信息
            np.array([
                self.current_time,           # 0: 全局时间
                min_psnr,                   # 1: 当前最小PSNR
                avg_psnr,                   # 2: 当前平均PSNR
                distance_to_start,          # 3: 到起点距离
                self.current_cycle,         # 4: 当前周期数
                self.cycle_step,           # 5: 周期内步数
                cycle_progress,            # 6: 周期进度
                float(exploration_phase),   # 7: 探索阶段标志
                float(self.returning_to_start), # 8: 返航状态
                cycle_psnr_std             # 9: 周期PSNR不平衡度
            ])
        ])
        
        return state.astype(np.float32)
    
    def step(self, action):
        # =============== 修正的运动模型 ===============
        target_yaw, target_pitch, target_speed = action
        
        # 获取当前状态
        x, y, z, current_yaw, current_pitch, current_speed, _ = self.uav_state
        
        # 限制角度变化率
        yaw_change = np.clip(target_yaw - current_yaw, 
                            -self.max_yaw_rate * self.time_slot,
                            self.max_yaw_rate * self.time_slot)
        pitch_change = np.clip(target_pitch - current_pitch,
                            -self.max_pitch_rate * self.time_slot,
                            self.max_pitch_rate * self.time_slot)
        
        # 更新姿态
        new_yaw = current_yaw + yaw_change
        new_pitch = current_pitch + pitch_change

        # 强制最小速度约束 - 防止完全不动
        min_speed = 1.0  # 最小速度1m/s
        new_speed = np.clip(target_speed, min_speed, self.max_horizontal_speed)
        
        # 修正的速度分量计算 - 使用标准航空坐标系
        # yaw: 0度指向+X轴(东)，逆时针为正
        # pitch: 0度为水平，向上为正
        v_x = new_speed * np.cos(new_pitch) * np.cos(new_yaw)  # 修正：cos(yaw)
        v_y = new_speed * np.cos(new_pitch) * np.sin(new_yaw)  # 修正：sin(yaw)
        v_z = new_speed * np.sin(new_pitch)
        
        # 直接计算位移
        dx = v_x * self.time_slot
        dy = v_y * self.time_slot
        dz = v_z * self.time_slot
        
        # 更新位置
        x_new = x + dx
        y_new = y + dy
        z_new = z + dz

        

        # =============== 边界检查和约束 ===============
        boundary_violation = False
        
        # 先检查边界违规（在clip之前）
        if (x_new < self.x_min or x_new > self.x_max or 
            y_new < self.y_min or y_new > self.y_max or 
            z_new < self.z_min or z_new > self.z_max):
            boundary_violation = True

        # 应用边界限制（硬约束）
        x_new = np.clip(x_new, self.x_min, self.x_max)
        y_new = np.clip(y_new, self.y_min, self.y_max)
        z_new = np.clip(z_new, self.z_min, self.z_max)

        # 更新UAV状态：[x, y, z, yaw, pitch, speed, roll]
        self.uav_state = np.array([x_new, y_new, z_new, new_yaw, new_pitch, new_speed, self.roll])

        # =============== 能耗计算（含姿态控制） ===============
        # 计算推进能耗（基于总速度）
        total_speed = np.sqrt(v_x**2 + v_y**2 + v_z**2)
        
        propulsion_power = calculate_propulsion_energy(
            P0=self.blade_power, 
            U_tip=self.tip_speed, 
            d0=self.drag_ratio, 
            rho=self.air_density, 
            s=self.rotor_solidity, 
            G=self.rotor_area, 
            P1=self.induced_power, 
            v0=self.hover_speed, 
            v_m_u=total_speed
        )

        # 推进能耗
        propulsion_energy = propulsion_power * self.time_slot
        model_energy = 10 * self.time_slot
        step_energy = propulsion_energy + model_energy
        self.energy -= step_energy

        # =============== 用户PSNR计算和记录 ===============
        user_psnrs = []
        user_distances = []
        
        for k in range(self.num_users):
            dist = np.linalg.norm(self.uav_state[:3] - self.users[k])
            if dist < 1e-6:
                dist = 1e-6
            G_i = calculate_path_loss(dist, self.eta_loss, self.f_c)
            snr = calculate_snr(self.eirp, G_i, self.bandwidth, self.noise_sigma)
            psnr = test_caculate_psnr(snr=snr)
                
            user_psnrs.append(psnr)
            user_distances.append(dist)
            
            # 更新周期内PSNR累积和最大值
            self.cycle_user_psnr_sum[k] += psnr
            self.cycle_user_max_psnr[k] = max(self.cycle_user_max_psnr[k], psnr)
            
            # 更新总体PSNR累积和最大值
            self.total_user_psnr_sum[k] += psnr
            self.total_user_max_psnr[k] = max(self.total_user_max_psnr[k], psnr)
            
            # 更新与用户的最小距离
            self.cycle_user_min_distance[k] = min(self.cycle_user_min_distance[k], dist)
            
            # 更新服务时间（如果在服务范围内，通常是30米以内）
            if dist <= 30.0:
                self.cycle_user_service_time[k] += 1
        
        # 找到最小和平均PSNR值
        min_psnr = min(user_psnrs)
        avg_psnr = np.mean(user_psnrs)
        
        # 计算到起点的距离
        distance_to_start = np.linalg.norm(self.uav_state[:3] - self.start_position)

        # =============== 周期性运动状态管理 ===============
        # 判断是否应该开始返航
        exploration_steps = int(self.cycle_length * self.exploration_ratio)
        if not self.returning_to_start and self.cycle_step >= exploration_steps:
            self.returning_to_start = True
            print(f"Cycle {self.current_cycle}: Starting return to start at step {self.cycle_step}")
        
        # 检查是否返回起点完成周期
        if self.returning_to_start and distance_to_start <= self.return_threshold:
            # 完成一个周期
            self._complete_cycle()
            self.returning_to_start = False
            self.at_start_position = True
            
            print(f"✓ Cycle {self.current_cycle-1} completed! Back to start.")
        else:
            self.at_start_position = (distance_to_start <= self.return_threshold)

        # =============== 修正的奖励函数设计 ===============
        reward = 0.0
        
        # 修正：大幅降低基础存活奖励，避免躺平
        reward += 0.02  # 从0.1降低到0.02
        
        # 修正：增加运动激励 - 强制鼓励移动
        movement_reward = min(total_speed / self.max_horizontal_speed, 1.0) * 1.5  # 最多+1.5分
        reward += movement_reward
        
        # 修正：位置变化奖励 - 鼓励改变位置
        if hasattr(self, 'prev_position'):
            position_change = np.linalg.norm(self.uav_state[:3] - self.prev_position)
            if position_change > 0.3:  # 位置变化超过0.3米
                change_bonus = min(position_change / 3.0, 1.0)  # 最多+1分
                reward += change_bonus
        
        if not self.returning_to_start:
            # ========== 探索和PSNR最大化阶段 ==========
            
            # 修正：探索奖励 - 鼓励远离起点
            distance_to_start_current = np.linalg.norm(self.uav_state[:3] - self.start_position)
            if distance_to_start_current > 5.0:  # 距离起点5米以上
                exploration_bonus = min(distance_to_start_current / 25.0, 2.0)  # 最多+2分
                reward += exploration_bonus
            
            # 1. 即时PSNR奖励（核心目标）- 降低权重
            instant_psnr_reward = 0.0
            for k, psnr in enumerate(user_psnrs):
                normalized_psnr = min(psnr / self.max_psnr, 1.0)
                instant_psnr_reward += normalized_psnr * 2.0  # 从3.0降低到2.0
            reward += instant_psnr_reward
            
            # 2. PSNR改善奖励：奖励使用户PSNR提升的行为
            psnr_improvement_reward = 0.0
            for k, psnr in enumerate(user_psnrs):
                if hasattr(self, 'prev_user_psnrs') and self.prev_user_psnrs[k] > 0:
                    improvement = psnr - self.prev_user_psnrs[k]
                    if improvement > 0:  # PSNR提升
                        psnr_improvement_reward += min(improvement / self.max_psnr * 3.0, 0.5)
            reward += psnr_improvement_reward
            
            # 3. 均衡服务奖励：鼓励为所有用户提供均衡的服务
            if len(user_psnrs) > 1:
                min_max_ratio = min_psnr / max(user_psnrs) if max(user_psnrs) > 0 else 0
                balance_reward = min_max_ratio * 1.5  # 从2.5降低到1.5
                reward += balance_reward
                
                psnr_variance = np.var(user_psnrs)
                if psnr_variance > 0:
                    balance_bonus = min(8.0 / (1 + psnr_variance), 1.0)  # 从1.5降低到1.0
                    reward += balance_bonus
            
            # 4. 覆盖奖励：鼓励接近服务不足的用户
            coverage_reward = 0.0
            for k in range(self.num_users):
                current_avg_psnr = self.cycle_user_psnr_sum[k] / max(self.cycle_step, 1)
                if current_avg_psnr < self.max_psnr * 0.6:  # 低于60%最大PSNR
                    proximity_factor = max(0, (25.0 - user_distances[k]) / 25.0)  # 25米范围内
                    coverage_reward += proximity_factor * 1.0  # 从1.5降低到1.0
            reward += min(coverage_reward, 3.0)  # 从4.0降低到3.0
            
            # 5. 多样性探索奖励：鼓励访问不同区域服务不同用户
            exploration_reward = 0.0
            services_count = sum(1 for d in user_distances if d <= 20.0)  # 20米内的用户数
            if services_count >= 2:  # 同时服务多个用户
                exploration_reward = min(services_count * 0.6, 2.0)  # 从2.5降低到2.0
            reward += exploration_reward
            
        else:
            # ========== 返航阶段 ==========
            
            # 6. 返航激励（强化）
            if distance_to_start > 1.0:
                return_reward = min(120.0 / distance_to_start, 12.0)  # 从15.0降低到12.0
                reward += return_reward
            else:
                reward += 12.0  # 从15.0降低到12.0
                
            # 7. 返航进步奖励
            progress = self.prev_distance_to_start - distance_to_start
            if progress > 0:  # 只奖励靠近行为
                progress_bonus = min(progress * 4.0, 3.0)  # 从5.0降低到3.0
                reward += progress_bonus
        
        # ========== 周期完成奖励 ==========
        if self.at_start_position and self.cycle_step > 0:
            # 8. 周期完成基础奖励
            cycle_completion_reward = 20.0  # 从25.0降低到20.0
            reward += cycle_completion_reward
            
            # 9. 周期PSNR性能奖励
            cycle_avg_psnr = np.mean(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
            cycle_max_psnr = np.mean(self.cycle_user_max_psnr)
            
            # 基于平均PSNR的性能奖励
            avg_psnr_performance = min(cycle_avg_psnr / self.max_psnr, 1.0)
            avg_psnr_bonus = avg_psnr_performance * 12.0  # 从15.0降低到12.0
            reward += avg_psnr_bonus
            
            # 基于最大PSNR的性能奖励  
            max_psnr_performance = min(cycle_max_psnr / self.max_psnr, 1.0)
            max_psnr_bonus = max_psnr_performance * 8.0  # 保持8.0
            reward += max_psnr_bonus
            
            # 10. 用户覆盖均衡奖励
            psnr_averages = self.cycle_user_psnr_sum / max(self.cycle_step, 1)
            psnr_std = np.std(psnr_averages)
            balance_factor = max(0, (self.max_psnr * 0.15 - psnr_std) / (self.max_psnr * 0.15))
            balance_bonus = balance_factor * 6.0  # 从8.0降低到6.0
            reward += balance_bonus
            
            # 11. 服务覆盖度奖励
            well_served_users = sum(1 for avg_psnr in psnr_averages if avg_psnr > self.max_psnr * 0.5)
            coverage_ratio = well_served_users / self.num_users
            coverage_bonus = coverage_ratio * 4.0  # 从5.0降低到4.0
            reward += coverage_bonus
            
            if self.current_time % 20 == 0:
                print(f"  Cycle rewards - Avg PSNR: {avg_psnr_bonus:.2f}, Max PSNR: {max_psnr_bonus:.2f}")
                print(f"  Balance: {balance_bonus:.2f}, Coverage: {coverage_bonus:.2f}")


        # =============== 飞行平滑度奖励 ===============
        # 记录上一步动作
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.array(action)
        action = np.array(action)
        # 计算动作变化幅度
        action_delta = np.abs(action - self.prev_action)
        # 惩罚yaw、pitch、speed变化过大（可调权重）
        smoothness_penalty = np.sum(action_delta / np.array([np.pi, np.pi/4, self.max_horizontal_speed]))
        reward -= smoothness_penalty * 2.0  # 权重2.0可根据实际调整

        # 更新历史动作
        self.prev_action = action.copy()
        
        # ========== 惩罚项 ==========
        # 修正：强化停留惩罚
        if total_speed < 0.5:  # 几乎不动
            reward -= 1.0  # 停留惩罚
            
        # 12. 边界违规惩罚
        if boundary_violation:
            reward -= 2.0  # 从3.0降低到2.0
            
        # 13. 能耗惩罚（轻微）
        energy_ratio = self.energy / self.max_energy
        if energy_ratio < 0.2:  # 能量不足20%时惩罚
            energy_penalty = (0.2 - energy_ratio) * 3.0  # 从5.0降低到3.0
            reward -= energy_penalty
        
        # ========== 最终奖励处理 ==========
        final_reward = np.clip(reward, -8.0, 50.0)  # 从60.0降低到50.0
        
        # 更新历史变量
        self.prev_distance_to_start = distance_to_start
        self.prev_user_psnrs = np.array(user_psnrs)
        self.prev_position = self.uav_state[:3].copy()  # 修正：更新位置历史
        self.current_time += 1
        self.cycle_step += 1

        # 终止条件：只有能量耗尽或超时才终止
        terminated = False  # 周期性运动不主动终止
        truncated = (self.current_time >= self.max_time or self.energy <= 0)
        
        # 终止时的额外奖励/惩罚（基于整体性能）
        if truncated:
            overall_avg_psnr = np.mean(self.total_user_psnr_sum / max(self.current_time, 1))
            overall_max_psnr = np.mean(self.total_user_max_psnr)
            
            if overall_avg_psnr > self.max_psnr * 0.7:
                final_reward += 25.0  # 从30.0降低到25.0
                print(f"✅ Excellent service! Overall avg PSNR: {overall_avg_psnr:.2f}")
            elif overall_avg_psnr > self.max_psnr * 0.5:
                final_reward += 15.0  # 从20.0降低到15.0
                print(f"✓ Good service! Overall avg PSNR: {overall_avg_psnr:.2f}")
            elif overall_avg_psnr > self.max_psnr * 0.3:
                final_reward += 8.0  # 从10.0降低到8.0
                print(f"○ Fair service! Overall avg PSNR: {overall_avg_psnr:.2f}")
            else:
                final_reward -= 5.0  # 从-10.0提高到-5.0
                print(f"⚠ Poor service: {overall_avg_psnr:.2f}")

        # 修正：增强调试信息（每10步打印一次）
        if self.current_time % 10 == 0:
            if self.returning_to_start:
                status = f"Returning (dist: {distance_to_start:.1f}m)"
            else:
                status = f"Exploring (cycle {self.current_cycle}, step {self.cycle_step})"
            
            print(f"Step {self.current_time}: {status}")
            print(f"  Action: yaw={target_yaw:.3f}, pitch={target_pitch:.3f}, speed={target_speed:.3f}")
            print(f"  Position: ({x_new:.1f}, {y_new:.1f}, {z_new:.1f})")
            print(f"  Velocity: vx={v_x:.3f}, vy={v_y:.3f}, vz={v_z:.3f}")
            print(f"  Total speed: {total_speed:.3f}")
            print(f"  Position change: {np.linalg.norm(self.uav_state[:3] - self.prev_position):.3f}")
            print(f"  PSNR - Min: {min_psnr:.2f}, Avg: {avg_psnr:.2f}")
            print(f"  Reward breakdown: base=0.02, move={movement_reward:.2f}, total={final_reward:.2f}")
            
            if total_speed < 0.1:
                print(f"  ⚠️ WARNING: UAV is nearly stationary!")
            
            # 显示周期内PSNR表现
            if self.cycle_step > 0:
                cycle_avg = np.mean(self.cycle_user_psnr_sum / self.cycle_step)
                cycle_max = np.mean(self.cycle_user_max_psnr)
                cycle_std = np.std(self.cycle_user_psnr_sum / self.cycle_step)
                print(f"  Cycle PSNR - Avg: {cycle_avg:.2f}, Max: {cycle_max:.2f}, Std: {cycle_std:.2f}")
                    
        return self._get_state(), final_reward, terminated, truncated, {}
    
    def _complete_cycle(self):
        """完成一个周期的处理"""
        # 计算周期性能指标
        cycle_avg_psnr = np.mean(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
        cycle_max_psnr = np.mean(self.cycle_user_max_psnr)
        cycle_min_psnr = np.min(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
        cycle_psnr_std = np.std(self.cycle_user_psnr_sum / max(self.cycle_step, 1))
        
        # 记录周期历史
        cycle_info = {
            'cycle': self.current_cycle,
            'duration': self.cycle_step,
            'avg_psnr': cycle_avg_psnr,
            'max_psnr': cycle_max_psnr,
            'min_psnr': cycle_min_psnr,
            'psnr_std': cycle_psnr_std,
            'user_avg_psnrs': (self.cycle_user_psnr_sum / max(self.cycle_step, 1)).copy(),
            'user_max_psnrs': self.cycle_user_max_psnr.copy(),
            'user_min_distances': self.cycle_user_min_distance.copy(),
            'user_service_times': self.cycle_user_service_time.copy()
        }
        self.cycle_history.append(cycle_info)
        
        print(f"Cycle {self.current_cycle} Summary:")
        print(f"  Duration: {self.cycle_step} steps")
        print(f"  PSNR Performance - Avg: {cycle_avg_psnr:.2f}, Max: {cycle_max_psnr:.2f}, Min: {cycle_min_psnr:.2f}")
        print(f"  PSNR Balance - Std: {cycle_psnr_std:.2f}")
        print(f"  Per-user performance:")
        for k in range(self.num_users):
            avg_psnr = self.cycle_user_psnr_sum[k] / max(self.cycle_step, 1)
            max_psnr = self.cycle_user_max_psnr[k]
            min_dist = self.cycle_user_min_distance[k]
            service_ratio = self.cycle_user_service_time[k] / max(self.cycle_step, 1)
            print(f"    User {k}: avg={avg_psnr:.2f}, max={max_psnr:.2f}, min_dist={min_dist:.1f}m, service={service_ratio:.2f}")
        
        # 重置周期相关变量
        self.current_cycle += 1
        self.cycle_step = 0
        
        # 重置周期PSNR记录
        self.cycle_user_psnr_sum = np.zeros(self.num_users)
        self.cycle_user_max_psnr = np.zeros(self.num_users)
        self.cycle_user_min_distance = np.full(self.num_users, float('inf'))
        self.cycle_user_service_time = np.zeros(self.num_users)