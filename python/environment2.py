import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections


class ActionSpace(spaces.Dict):
    def __init__(self):
        u_l=spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        u_r=spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        super().__init__(spaces.Dict({'u':u_l, 'v':u_r}))
    
class ObservationSpace(spaces.Dict):
    def __init__(self):
        theta_l=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        theta_r=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        theta_1=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        theta_2=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_l=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_r=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_1=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_2=spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        super().__init__(spaces.Dict({'theta_l':theta_l, 'theta_r':theta_r, 'theta_1':theta_1, 'theta_2':theta_2,'d_theta_l':d_theta_l, 'd_theta_r':d_theta_r, 'd_theta_1':d_theta_1, 'd_theta_2':d_theta_2}))

class Env_2wheels(gym.Env):
    def __init__(self):
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.state = collections.OrderedDict()
        self.reset()

    def reset(self):
        self.state:collections.OrderedDict=self.observation_space.sample()
        self.state['d_theta_l']=np.zeros((1,), dtype=np.float32)
        self.state['d_theta_r']=np.zeros((1,), dtype=np.float32)
        self.state['d_theta_1']=np.zeros((1,), dtype=np.float32)
        self.state['d_theta_2']=np.zeros((1,), dtype=np.float32)

    
    def init_parameters():
        # Parameters
        m_1 = 0.9    # 车体质量
        m_2 = 0.1    # 摆杆质量
        r = 0.0335   # 车轮半径
        L_1 = 0.126  # 车体的长度
        L_2 = 0.390  # 摆杆的长度
        l_1 = L_1 / 2  # 车体的质心到转轴1的距离
        l_2 = L_2 / 2  # 摆杆的质心到转轴2的距离
        g = 9.8
        I_1 = (1 / 12) * m_1 * L_1 ** 2  # 车体绕其质心转动时的转动惯量
        I_2 = (1 / 12) * m_2 * L_2 ** 2  # 摆杆绕其质心转动时的转动惯量

        # p matrix
        p_11 = 1
        p_12 = 0
        p_13 = 0
        p_14 = 0
        p_21 = 0
        p_22 = 1
        p_23 = 0
        p_24 = 0
        p_31 = (r / 2) * (m_1 * l_1 + m_2 * L_1)
        p_32 = (r / 2) * (m_1 * l_1 + m_2 * L_1)
        p_33 = m_1 * l_1 ** 2 + m_2 * L_1 ** 2 + I_1
        p_34 = m_2 * L_1 * l_2
        p_41 = (r / 2) * m_2 * l_2
        p_42 = (r / 2) * m_2 * l_2
        p_43 = m_2 * L_1 * l_2
        p_44 = m_2 * l_2 ** 2 + I_2

        p = np.array([
            [p_11, p_12, p_13, p_14],
            [p_21, p_22, p_23, p_24],
            [p_31, p_32, p_33, p_34],
            [p_41, p_42, p_43, p_44]
        ])

        # q matrix
        q_11 = 0
        q_12 = 0
        q_13 = 0
        q_14 = 0
        q_15 = 0
        q_16 = 0
        q_17 = 0
        q_18 = 0
        q_19 = 1
        q_110 = 0
        q_21 = 0
        q_22 = 0
        q_23 = 0
        q_24 = 0
        q_25 = 0
        q_26 = 0
        q_27 = 0
        q_28 = 0
        q_29 = 0
        q_210 = 1
        q_31 = 0
        q_32 = 0
        q_33 = (m_1 * l_1 + m_2 * L_1) * g
        q_34 = 0
        q_35 = 0
        q_36 = 0
        q_37 = 0
        q_38 = 0
        q_39 = 0
        q_310 = 0
        q_41 = 0
        q_42 = 0
        q_43 = 0
        q_44 = m_2 * g * l_2
        q_45 = 0
        q_46 = 0
        q_47 = 0
        q_48 = 0
        q_49 = 0
        q_410 = 0

        q = np.array([
            [q_11, q_12, q_13, q_14, q_15, q_16, q_17, q_18, q_19, q_110],
            [q_21, q_22, q_23, q_24, q_25, q_26, q_27, q_28, q_29, q_210],
            [q_31, q_32, q_33, q_34, q_35, q_36, q_37, q_38, q_39, q_310],
            [q_41, q_42, q_43, q_44, q_45, q_46, q_47, q_48, q_49, q_410]
        ])

        # Calculate results
        temp = np.linalg.inv(p) @ q

        # A matrix
        A = np.array([  
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        A = np.vstack((A, temp[:, :8]))  # 8 x 8

        # B matrix
        B = np.array([  # 4 x 2
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        B = np.vstack((B, temp[:, 8:10]))  # 8 x 2

        # C matrix
        C = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # D matrix
        D = np.zeros((4, 2))

        params = {
            'dt': 0.01,
            'm_1': 0.9,     # 车体质量 (kg)
            'm_2': 0.1,     # 摆杆质量 (kg)
            'r': 0.0335,    # 车轮半径 (m)
            'L_1': 0.126,   # 车体长度 (m)
            'L_2': 0.390,   # 摆杆长度 (m)
            'l_1': 0.126/2, # 车体质心到转轴距离 (m)
            'l_2': 0.390/2, # 摆杆质心到转轴距离 (m)
            'g': 9.8,       # 重力加速度 (m/s^2)
            'I_1': (1/12)*0.9*(0.126**2),  # 车体转动惯量
            'I_2': (1/12)*0.1*(0.390**2),   # 摆杆转动惯量
            'A': A,
            'B': B,
            'theta_LR': 0,  # 车轮转过的角度
            'theta_1': 0,   # 车体的倾角
            'theta_2': 0,   # 摆杆的倾角
            'dtheta_LR': 0, # 车轮的角速度
            'dtheta_1': 0,  # 倾角的角速度
            'dtheta_2': 0,  # 摆杆倾角的角速度
            'u_LR': 0,      # 车轮的加速度
            'bins': 6,
            'LR_range': 100,
            '12_range': 10,
            'u_LR_range': 10,
            'u_sample_rate': 10
        }

        return params
    def step(self,action):

        dt=1/100
        '''动作的间隔时间'''
        params=self.init_parameters()#引用了同学的代码
        '''初始化参数'''
        
        next_state = self.state.copy()
        # 计算下一个状态
        # 将状态转换为列向量
        state_value=np.array(self.state.values()).reshape(-1,1)
        # 将动作转换为列向量
        action_value=np.array(action.values()).reshape(-1,1)
        
        delta_state:np.array=params['A'] @ state_value + params['B'] @ action_value

        for key,value in next_state.items():
            next_state[key]=value+delta_state*dt
        
        reward=reward(next_state,action)
        done = self.is_done(next_state)
        self.state=next_state
        return next_state, reward, done, False, {}
    

    def reward(self, state, action):
        # 权重系数
        w_1 = 1.0  # 角度惩罚权重
        w_2 = 0.5  # 角速度惩罚权重
        w_3 = 0.1  # 控制输入惩罚权重

        # 提取状态和动作
        theta_1 = state['theta_1']
        theta_2 = state['theta_2']
        d_theta_1 = state['d_theta_1']
        d_theta_2 = state['d_theta_2']
        u_L = action['u_l']
        u_R = action['u_r']

        # 计算奖励
        reward = -w_1 * (theta_1 ** 2 + theta_2 ** 2) - w_2 * (d_theta_1 ** 2 + d_theta_2 ** 2) - w_3 * (u_L ** 2 + u_R ** 2)
        return reward
    
    def is_done(self, state):
        # 判断是否终止
        theta_1 = state['theta_1']
        theta_2 = state['theta_2']
        if abs(theta_1) > np.pi / 4 or abs(theta_2) > np.pi / 4:
            return True
        return False








    
        






