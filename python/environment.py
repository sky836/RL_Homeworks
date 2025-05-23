import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt
import numpy as np
import collections


class ActionSpace(spaces.Dict):
    def __init__(self):
        u_lr = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        super().__init__({'u_lr': u_lr})


class ObservationSpace(spaces.Dict):
    def __init__(self):
        theta_lr = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        theta_1 = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        theta_2 = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_lr = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_1 = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        d_theta_2 = spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        super().__init__(collections.OrderedDict({'theta_lr': theta_lr, 'theta_1': theta_1, 'theta_2': theta_2,
                          'd_theta_lr': d_theta_lr, 'd_theta_1': d_theta_1, 'd_theta_2': d_theta_2}))


class DiscreteEnv(gym.Env):
    def __init__(self):
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.state = collections.OrderedDict()
        self.reset()
        self.fig = None
        self.ax = None
        self.params = {
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
            'bins': 20,
            'LR_range': 5*np.pi,
            'd_LR_range': 5,
            'range_1': np.pi/2,
            'range_2': np.pi,
            'd_range_12': 5,
            'u_LR_range': 5,
            'u_sample_rate': 5,
            'action_bins': 14
        }
        dt=self.params['dt']
        # 参数
        m_1 = self.params['m_1']
        m_2 = self.params['m_2']
        r = self.params['r']
        L_1 = self.params['L_1']
        L_2 = self.params['L_2']
        l_1 = self.params['l_1']
        l_2 = self.params['l_2']
        g = self.params['g']
        I_1 = self.params['I_1']
        I_2 = self.params['I_2']
        
        # p矩阵
        p_11 = 1
        p_12 = 0
        p_13 = 0
        p_14 = 0
        p_21 = 0
        p_22 = 1
        p_23 = 0
        p_24 = 0
        p_31 = (r/2)*(m_1*l_1 + m_2*L_1)
        p_32 = (r/2)*(m_1*l_1 + m_2*L_1)
        p_33 = m_1*l_1**2 + m_2*L_1**2 + I_1
        p_34 = m_2*L_1*l_2
        p_41 = (r/2)*m_2*l_2
        p_42 = (r/2)*m_2*l_2
        p_43 = m_2*L_1*l_2
        p_44 = m_2*l_2**2 + I_2
        p = np.matrix([[p_11, p_12, p_13, p_14],
                       [p_21, p_22, p_23, p_24],
                       [p_31, p_32, p_33, p_34],
                       [p_41, p_42, p_43, p_44]])
        
        # q矩阵
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
        q_33 = (m_1*l_1 + m_2*L_1)*g
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
        q_44 = m_2*g*l_2
        q_45 = 0
        q_46 = 0
        q_47 = 0
        q_48 = 0
        q_49 = 0
        q_410 = 0
        q = np.matrix([[q_11, q_12, q_13, q_14, q_15, q_16, q_17, q_18, q_19, q_110],
                       [q_21, q_22, q_23, q_24, q_25, q_26, q_27, q_28, q_29, q_210],
                       [q_31, q_32, q_33, q_34, q_35, q_36, q_37, q_38, q_39, q_310],
                       [q_41, q_42, q_43, q_44, q_45, q_46, q_47, q_48, q_49, q_410]])

        # 计算结果
        temp = p.I @ q
        self.A = np.append([[0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1]], temp[:, 0:8], axis=0)
        self.B = np.append([[0, 0], [0, 0], [0, 0], [0, 0]], temp[:, 8:10], axis=0)

    def reset(self):
        self.state = self.observation_space.sample()
        self.state['d_theta_lr'] = np.zeros((1,), dtype=np.float32)
        self.state['d_theta_1'] = np.zeros((1,), dtype=np.float32)
        self.state['d_theta_2'] = np.zeros((1,), dtype=np.float32)
        return self.state

    def step(self, action):
    
        # 确保状态值是数值类型
        state_vec = np.array([
            float(self.state['theta_lr']), 
            float(self.state['theta_lr']), 
            float(self.state['theta_1']), 
            float(self.state['theta_2']),
            float(self.state['d_theta_lr']),
            float(self.state['d_theta_lr']),
            float(self.state['d_theta_1']),
            float(self.state['d_theta_2'])
        ])
        
        # 确保动作值是数值类型
        action_vec = np.array([float(action['u_lr']), float(action['u_lr'])])
        
        # 计算下一个状态
        next_state_vec = (np.matmul(self.A, state_vec) + np.matmul(self.B, action_vec))*self.params['dt']+state_vec
        next_state_vec = next_state_vec.reshape(-1, 1)
        
        next_state = collections.OrderedDict({
            'theta_lr': np.array([next_state_vec[0]], dtype=np.float32),
            'theta_1': np.array([next_state_vec[2]], dtype=np.float32),
            'theta_2': np.array([next_state_vec[3]], dtype=np.float32),
            'd_theta_lr': np.array([next_state_vec[4]], dtype=np.float32),
            'd_theta_1': np.array([next_state_vec[6]], dtype=np.float32),
            'd_theta_2': np.array([next_state_vec[7]], dtype=np.float32)
        })

        self.state = next_state
        
        reward = self.get_reward(next_state, action)
        terminated = self.is_terminated(next_state)
        
        return next_state, reward, terminated, {}

    def get_reward(self, state, action):
        l1 = self.params['L_1']
        l2 = self.params['L_2']
        r = self.params['r']
        theta_LR = float(state['theta_lr'])
        theta_1 = float(state['theta_1'])
        theta_2 = float(state['theta_2'])
        d_theta_lr = float(state['d_theta_lr'])
        d_theta_1 = float(state['d_theta_1'])
        d_theta_2 = float(state['d_theta_2'])
        l_now = l1*np.cos(theta_1) + l2*np.cos(theta_2) + 2*r
        L = l1 + l2 + 2*r

        # reward = -(100 * np.abs(theta_1) + 100 * np.abs(theta_2) + 2 * np.abs(theta_LR))
        if l_now >= L*0.7:
            healthy_reward = 10
        else:
            healthy_reward = -10
        velocity_penalty = 0.001*d_theta_1 + 0.0001*d_theta_2
        distance_penalty = 0.01*np.abs(theta_LR * r)
        theta_penalty = np.abs(theta_1) + np.abs(theta_2)
        reward = healthy_reward - velocity_penalty - distance_penalty
        return reward

    def is_terminated(self, state):
        l1 = self.params['L_1']
        l2 = self.params['L_2']
        r = self.params['r']
        theta_LR = float(state['theta_lr'][0])
        theta_1 = float(state['theta_1'][0])
        theta_2 = float(state['theta_2'][0])
        l_now = l1*np.cos(theta_1) + l2*np.cos(theta_2) + 2*r
        min_l = l2 + 2*r + l1*0.1

        if l_now < min_l:
            return True
        return False

    def render(self):
        """Render the current state of the environment with two distinct poles"""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            
            # 动态设置坐标范围（基于摆杆长度）
            max_length = self.params['L_1'] + self.params['L_2']
            self.ax.set_xlim(-max_length - 0.5, max_length + 0.5)
            self.ax.set_ylim(-max_length * 0.2, max_length * 1.5)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            
            # 初始化绘图对象
            self.cart = plt.Rectangle((-0.2, -0.1), 0.4, 0.2, fc='blue')
            self.wheel_left = plt.Circle((-0.15, -0.15), 0.05, fc='black')
            self.wheel_right = plt.Circle((0.15, -0.15), 0.05, fc='black')
            
            # 第一段摆杆（红色）
            self.pole1, = self.ax.plot([0, 0], [0, 0], 'r-', lw=3)
            # 第二段摆杆（绿色）
            self.pole2, = self.ax.plot([0, 0], [0, 0], 'g-', lw=3)
            # 铰链标记（黑色圆点）
            self.joint = plt.Circle((0, 0), 0.03, fc='black')
            
            self.ax.add_patch(self.cart)
            self.ax.add_patch(self.wheel_left)
            self.ax.add_patch(self.wheel_right)
            self.ax.add_patch(self.joint)

        # 确保状态值是数值类型
        theta_LR = float(self.state['theta_lr'][0])
        theta_1 = float(self.state['theta_1'][0])
        theta_2 = float(self.state['theta_2'][0])
        
        # 小车位置（简化模型）
        cart_x = theta_LR * self.params['r']
        
        # 更新小车和轮子位置
        self.cart.set_xy((cart_x - 0.2, -0.1))
        self.wheel_left.center = (cart_x - 0.15, -0.15)
        self.wheel_right.center = (cart_x + 0.15, -0.15)
        
        # 计算第一段摆杆的末端位置（铰链点）
        pole1_x = cart_x + np.sin(theta_1) * self.params['L_1']
        pole1_y = np.cos(theta_1) * self.params['L_1']
        
        # 计算第二段摆杆的末端位置（基于第一段的角度叠加）
        pole2_x = pole1_x + np.sin(theta_2) * self.params['L_2']
        pole2_y = pole1_y + np.cos(theta_2) * self.params['L_2']
        
        # 更新摆杆和铰链位置
        self.pole1.set_data([cart_x, pole1_x], [0, pole1_y])  # 第一段（红）
        self.pole2.set_data([pole1_x, pole2_x], [pole1_y, pole2_y])  # 第二段（绿）
        self.joint.center = (pole1_x, pole1_y)  # 铰链标记
        
        # 强制刷新图像
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


if __name__ == "__main__":
    env = DiscreteEnv()
    state = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        next_state, reward, terminated, _ = env.step(action)
        env.render()
        if terminated:
            state = env.reset()
    env.close()