from environment import *
import numpy as np
import collections
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings('ignore')

class DiscreteEnvWrapper:
    def __init__(self, env:Environment, bins=10):
        self.env = env
        self.bins = env.params['bins']
        self.a_bins = env.params['action_bins']
        LR_range = self.env.params['LR_range']
        range_1 = self.env.params['range_1']
        range_2 = self.env.params['range_2']
        u_LR_range = self.env.params['u_LR_range']
        d_LR_range = self.env.params['d_LR_range']
        d_range_12 = self.env.params['d_range_12']
        
        # 离散化 theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
        self.theta_lr_bins = np.linspace(-LR_range, LR_range, self.bins)
        self.theta_1_bins = np.linspace(-range_1, range_1, self.bins)
        self.theta_2_bins = np.linspace(-range_2, range_2, self.bins)
        self.d_theta_lr_bins = np.linspace(-d_LR_range, d_LR_range, self.bins)  # 假设角速度范围 [-5, 5]
        self.d_theta_1_bins = np.linspace(-d_range_12, d_range_12, self.bins)
        self.d_theta_2_bins = np.linspace(-d_range_12, d_range_12, self.bins)
        
        # 离散化动作空间（假设 u_lr ∈ [-1, 1]）
        self.action_bins = np.linspace(-u_LR_range, u_LR_range, self.a_bins)
        
    def discretize_state(self, state):
        """将连续状态离散化为整数索引"""
        theta_lr_idx = np.digitize(state['theta_lr'], self.theta_lr_bins) - 1
        theta_1_idx = np.digitize(state['theta_1'], self.theta_1_bins) - 1
        theta_2_idx = np.digitize(state['theta_2'], self.theta_2_bins) - 1
        d_theta_lr_idx = np.digitize(state['d_theta_lr'], self.d_theta_lr_bins) - 1
        d_theta_1_idx = np.digitize(state['d_theta_1'], self.d_theta_1_bins) - 1
        d_theta_2_idx = np.digitize(state['d_theta_2'], self.d_theta_2_bins) - 1
        return (theta_lr_idx, theta_1_idx, theta_2_idx, d_theta_lr_idx, d_theta_1_idx, d_theta_2_idx)
    
    def discretize_action(self, action):
        """将连续动作离散化为整数索引"""
        u_lr_idx = np.digitize(action['u_lr'], self.action_bins) - 1
        return u_lr_idx
    
    def get_action_from_idx(self, action_idx):
        """从离散动作索引恢复连续动作"""
        u_lr = self.action_bins[action_idx]
        return {'u_lr': np.array([u_lr], dtype=np.float32)}
    
    def get_continuous_state(self, s):
        """将离散状态索引 s 转换为连续状态值"""
        theta_lr_idx, theta_1_idx, theta_2_idx, d_theta_lr_idx, d_theta_1_idx, d_theta_2_idx = s
        
        # 计算各维度的连续值（取对应区间的中点）
        theta_lr = self.theta_lr_bins[theta_lr_idx]
        theta_1 = self.theta_1_bins[theta_1_idx]
        theta_2 = self.theta_2_bins[theta_2_idx]
        d_theta_lr = self.d_theta_lr_bins[d_theta_lr_idx]
        d_theta_1 = self.d_theta_1_bins[d_theta_1_idx]
        d_theta_2 = self.d_theta_2_bins[d_theta_2_idx]
        
        # 返回与原始环境状态格式一致的 OrderedDict
        return collections.OrderedDict({
            'theta_lr': np.array([theta_lr], dtype=np.float32),
            'theta_1': np.array([theta_1], dtype=np.float32),
            'theta_2': np.array([theta_2], dtype=np.float32),
            'd_theta_lr': np.array([d_theta_lr], dtype=np.float32),
            'd_theta_1': np.array([d_theta_1], dtype=np.float32),
            'd_theta_2': np.array([d_theta_2], dtype=np.float32)
        })
def policy_iteration(env_wrapper:DiscreteEnvWrapper, gamma=0.99, max_iter=1000, theta=1e-4):
    env = env_wrapper.env
    n_bins = env_wrapper.bins
    n_dims = 6  # theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
    n_states = n_bins ** n_dims  # 例如 10^6 = 1,000,000
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    
    # 初始化值函数和策略
    V = np.zeros(state_shape)
    policy = np.random.randint(0, n_actions, size=state_shape)
    
    for _ in range(max_iter):
        # 策略评估
        while True:
            delta = 0
            for s in np.ndindex(*state_shape):
                v = V[s]
                a = policy[s]
                action = env_wrapper.get_action_from_idx(a)
                
                # 执行动作，得到下一个状态和奖励
                env.state = env_wrapper.get_continuous_state(s)
                next_state, reward, done, _ = env.step(action)
                s_next = env_wrapper.discretize_state(next_state)
                
                # 更新值函数
                V[s] = reward + gamma * (0 if done else V[s_next].item())
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                print('delta:', delta)
                break    
        
        # 策略改进
        policy_stable = True
        for s in np.ndindex(*state_shape):
            old_action = policy[s]
            action_values = np.zeros(n_actions)
            
            for a in range(n_actions):
                action = env_wrapper.get_action_from_idx(a)
                env.state = env_wrapper.get_continuous_state(s)
                next_state, reward, done, _ = env.step(action)
                s_next = env_wrapper.discretize_state(next_state)
                action_values[a] = reward + gamma * (0 if done else V[s_next].item())
            
            # 选择最优动作
            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break

    
    return V, policy

def test_policy_iteration(env_wrapper:DiscreteWrapper,gamma=0.99, max_iter=1000, theta=1e-4):
    n_bins=env_wrapper.bins
    n_dims=len(env_wrapper.env.update_params.range)
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    V=np.zeros(state_shape)
    policy=np.random.randint(0, n_actions, size=state_shape)

    # for _ in range(max_iter):
    #     #策略评估
    #     while True:
    #         delta = 0
    #         for s in 
    #         if delta < theta:
    #             break
        

    
            
                


def value_iteration(env_wrapper, gamma=0.99, max_iter=1000, theta=1e-4):
    env = env_wrapper.env
    n_bins = env_wrapper.bins
    n_dims = 6  # theta_lr, theta_1, theta_2, d_theta_lr, d_theta_1, d_theta_2
    n_states = n_bins ** n_dims  # 例如 10^6 = 1,000,000
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims
    
    V = np.zeros(state_shape)
    
    for _ in range(max_iter):
        delta = 0
        for s in np.ndindex(*state_shape):
            v = V[s]
            max_value = -np.inf
            
            for a in range(n_actions):
                action = env_wrapper.get_action_from_idx(a)
                env.state = env_wrapper.get_continuous_state(s)
                next_state, reward, done, _ = env.step(action)
                s_next = env_wrapper.discretize_state(next_state)
                value = reward + gamma * (0 if done else V[s_next])
                if value > max_value:
                    max_value = value
            
            V[s] = max_value.item()
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # 提取最优策略
    policy = np.zeros(state_shape, dtype=int)
    for s in np.ndindex(*state_shape):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action = env_wrapper.get_action_from_idx(a)
            env.state = env_wrapper.get_continuous_state(s)
            next_state, reward, done, _ = env.step(action)
            s_next = env_wrapper.discretize_state(next_state)
            action_values[a] = reward + gamma * (0 if done else V[s_next].item())
        policy[s] = np.argmax(action_values)
    
    return V, policy


def q_learning(env_wrapper, gamma=0.01, alpha=0.5, epsilon=0.1, episodes=100000):
    env = env_wrapper.env
    n_bins = env_wrapper.bins
    n_dims = 6  # 状态维度
    n_actions = env_wrapper.a_bins
    state_shape = (n_bins,) * n_dims

    # 初始化 Q 表
    Q = np.zeros(state_shape + (n_actions,))

    for _ in range(episodes):
        # 初始化状态
        state = env.reset()
        s = env_wrapper.discretize_state(state)
        done = False

        while not done:
            # ε-greedy 选择动作
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = np.argmax(Q[s])

            # 执行动作
            action = env_wrapper.get_action_from_idx(a)
            next_state, reward, done, _ = env.step(action)
            s_next = env_wrapper.discretize_state(next_state)

            # Q-learning 更新
            Q_target = reward + gamma * np.max(Q[s_next]) if not done else reward
            Q[s + (a,)] += alpha * (Q_target - Q[s + (a,)])

            # 转移到下一个状态
            s = s_next

    # 从 Q 表提取最优策略
    policy = np.argmax(Q, axis=-1)
    return Q, policy

def show_res(history):
    # Plot test results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(5, 1, 1)
    plt.plot(history[:, 0])
    plt.title('Cart Position')
    plt.ylabel('x (m)')
    
    plt.subplot(5, 1, 2)
    plt.plot(history[:, 1])
    plt.title('first Pole Angle')
    plt.ylabel('theta (rad)')

    plt.subplot(5, 1, 3)
    plt.plot(history[:, 2])
    plt.title('Second Pole Angle')
    plt.ylabel('theta (rad)')
    
    plt.subplot(5, 1, 4)
    plt.plot(history[:, 3])
    plt.title('Control Force')
    plt.ylabel('F (N)')
    plt.xlabel('Time step')

    plt.subplot(5, 1, 5)
    plt.plot(history[:, 4])
    plt.title('Reward')
    plt.ylabel('v')
    plt.xlabel('Time step')
    
    plt.tight_layout()
    plt.show()


def test():
    env=Environment()
    observation_wrapper=DiscreteEnvWrapper(env, bins=10)  
    


if __name__ == "__main__":
    env = Environment()
    env_wrapper = DiscreteEnvWrapper(env, bins=10)  # 离散化为 5 bins
    
    # print("Running Policy Iteration...")
    # V_pi, policy_pi = policy_iteration(env_wrapper)
    # print("Policy Iteration Completed!")
    
    print("Running Q Learning...")
    begin_time = time.time()
    Q, policy_q = q_learning(env_wrapper)
    cost = time.time()-begin_time
    print('cost time:', cost)
    print("Q Learning Completed!")
    
    # print("\nRunning Value Iteration...")
    # V_vi, policy_vi = value_iteration(env_wrapper)
    # print("Value Iteration Completed!")
    
    # # 比较两种算法的策略是否一致
    # print("\nPolicy Difference:", np.sum(policy_pi != policy_vi))

    # V_pi = np.load('V_pi.npy')
    # policy_pi = np.load('policy_pi.npy')
    state = env.reset()
    print(env.state)
    l = 2000
    history = np.zeros((l, 5))  # [theta_LR, theta_1, theta_2, action, reward]
    for t in range(l):
        s = env_wrapper.discretize_state(state)
        action_idx =policy_q[s]
        # print(action_idx)
        action = env_wrapper.get_action_from_idx(action_idx)
        # print(action)
        next_state, reward, terminated, _ = env.step(action)
        # print(env.state)
        # 确保状态值是数值类型
        theta_LR = float(next_state['theta_lr'][0][0])
        theta_1 = float(next_state['theta_1'][0][0])
        theta_2 = float(next_state['theta_2'][0][0])
        action = float(action['u_lr'][0][0])
        reward = float(reward)
        history[t] = np.array([theta_LR, theta_1, theta_2, action, reward])
        env.render()
        if terminated:
            state = env.reset()
    env.close()
    show_res(history)