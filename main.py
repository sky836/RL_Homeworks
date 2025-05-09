import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class CartPoleTwoWheelsEnv(gym.Env):
    """Custom Gymnasium environment for your two-wheeled cart-pole system"""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(CartPoleTwoWheelsEnv, self).__init__()
        
        # Initialize parameters
        self.params = self._init_parameters()
        
        # State space: [theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, dtheta_2]
        self.state = None
        
        # Action space: acceleration for both wheels (same value for both)
        self.action_space = spaces.Box(
            low=-self.params['u_LR_range'],
            high=self.params['u_LR_range'],
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([
                -self.params['LR_range'], 
                -self.params['12_range'], 
                -self.params['12_range'],
                -self.params['u_LR_range'],
                -self.params['u_LR_range'],
                -self.params['u_LR_range']
            ]),
            high=np.array([
                self.params['LR_range'], 
                self.params['12_range'], 
                self.params['12_range'],
                self.params['u_LR_range'],
                self.params['u_LR_range'],
                self.params['u_LR_range']
            ]),
            dtype=np.float32
        )
        
        # Visualization
        self.fig = None
        self.ax = None
        
    def _init_parameters(self):
        """Initialize physical parameters (same as your code)"""
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
            'bins': 6,
            'LR_range': 100,
            '12_range': 10,
            'u_LR_range': 10,
            'u_sample_rate': 10
        }

        return params
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initial state with small random perturbation
        self.state = np.array([
            0,                      # theta_LR
            0.1 * self.np_random.uniform(-1, 1),  # theta_1
            0.1 * self.np_random.uniform(-1, 1),  # theta_2
            0,                      # dtheta_LR
            0,                      # dtheta_1
            0                       # dtheta_2
        ])
        
        # Return observation and info dict
        return self.state, {}
    
    def step(self, action):
        """Run one timestep of the environment's dynamics"""
        # Get current state
        theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, dtheta_2 = self.state
        
        # Apply action (same acceleration for both wheels)
        u = np.array([action[0], action[0]])  # Using same action for both wheels
        
        # Calculate next state using your dynamics
        pre_state = np.array([
            theta_LR, theta_LR, theta_1, theta_2, 
            dtheta_LR, dtheta_LR, dtheta_1, dtheta_2
        ])
        
        next_state = np.matmul(self.params['A'], pre_state) + np.matmul(self.params['B'], u)
        
        # Extract the 6-dimensional state we care about
        self.state = np.array([
            next_state[0],  # theta_LR
            next_state[2],  # theta_1
            next_state[3],  # theta_2
            next_state[4],  # dtheta_LR
            next_state[6],  # dtheta_1
            next_state[7]   # dtheta_2
        ])
        
        # Calculate reward
        reward = self._get_reward(self.state, action)
        
        # Check termination conditions
        terminated = self._is_terminated(self.state)
        
        # Truncation (not used here)
        truncated = False
        
        # Additional info
        info = {}
        
        return self.state, reward, terminated, truncated, info
    
    def _get_reward(self, state, action):
        """Calculate reward based on current state and action"""
        theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, dtheta_2 = state
        
        # Main penalty for angle deviation, secondary for position and control force
        reward = -(100 * np.abs(theta_1) + 100 * np.abs(theta_2) + 2 * np.abs(theta_LR))
        
        # Small penalty for large actions to encourage smoother control
        reward -= 0.01 * (action[0] ** 2)
        
        return reward
    
    def _is_terminated(self, state):
        """Check if episode should terminate"""
        theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, dtheta_2 = state
        
        # Episode terminates if angles exceed thresholds
        if (abs(theta_1) > self.params['12_range'] or 
            abs(theta_2) > self.params['12_range'] or
            abs(theta_LR) > self.params['LR_range']):
            return True
        
        return False
    
    def render(self):
        """Render the current state of the environment"""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-1, 1)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            
            # Initialize plot objects
            self.cart = plt.Rectangle((-0.2, -0.1), 0.4, 0.2, fc='blue')
            self.wheel_left = plt.Circle((-0.15, -0.15), 0.05, fc='black')
            self.wheel_right = plt.Circle((0.15, -0.15), 0.05, fc='black')
            self.pole, = self.ax.plot([0, 0], [0, 0.5], 'r-', lw=3)
            
            self.ax.add_patch(self.cart)
            self.ax.add_patch(self.wheel_left)
            self.ax.add_patch(self.wheel_right)
        
        theta_LR, theta_1, theta_2, _, _, _ = self.state
        
        # Cart position (simplified)
        cart_x = theta_LR * self.params['r']  # Convert wheel angle to position
        
        # Update cart position
        self.cart.set_xy((cart_x - 0.2, -0.1))
        self.wheel_left.center = (cart_x - 0.15, -0.15)
        self.wheel_right.center = (cart_x + 0.15, -0.15)
        
        # Update pole position (combining both angles)
        pole_x1 = cart_x
        pole_y1 = 0
        pole_x2 = cart_x + np.sin(theta_1 + theta_2) * self.params['L_2']
        pole_y2 = np.cos(theta_1 + theta_2) * self.params['L_2']
        self.pole.set_data([pole_x1, pole_x2], [pole_y1, pole_y2])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Clean up resources"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# Example of how to use the environment
if __name__ == "__main__":
    env = CartPoleTwoWheelsEnv()
    
    # Test with random actions
    obs, _ = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            obs, _ = env.reset()
    
    env.close()