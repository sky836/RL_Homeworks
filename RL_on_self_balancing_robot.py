import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from itertools import product

# def init_parameters():
#     """Initialize physical and RL parameters"""
#     params = {
#         'm_cart': 1.0,      # Cart mass (kg)
#         'm_pole': 0.1,      # Pole mass (kg)
#         'l': 0.5,          # Pole length (m)
#         'g': 9.81,         # Gravity (m/s^2)
#         'dt': 0.02,        # Time step (s)
#         'max_force': 10,   # Max control force (N)
#         'x_threshold': 2.4, # Cart position threshold (m)
#         'theta_threshold': 12 * np.pi/180, # Angle threshold (rad)
#         'x_bins': 11,       # Position bins
#         'dx_bins': 11,      # Velocity bins
#         'theta_bins': 11,   # Angle bins
#         'dtheta_bins': 11,   # Angular velocity bins
#         'theta_L': None,   # 左轮转过的角度
#         'theta_R': None,   # 右轮转过的角度
#         'theta_1': None,   # 车轮的倾角
#         'theta_2': None,   # 摆杆的倾角
#         'dtheta_L': None,  # 左轮的角速度
#         'dtheta_R': None,  # 右轮的角速度
#         'dtheta_1': None,  # 倾角的角速度
#         'dtheta_2': None,  # 摆杆倾角的角速度
#         'u_L': None,       # 左车轮的加速度
#         'u_R': None,       # 右车轮的加速度
#         'ddtheta_L': None,
#         'ddtheta_R': None,
#         'ddtheta_1': None,
#         'ddtheta_2': None
#     }
#     return params

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

def get_next_state(s, a, params):
    theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, d_theta_2  = s
    u = np.array([a, a])

    pre = np.array([theta_LR, theta_LR, theta_1, theta_2, dtheta_LR, dtheta_LR, dtheta_1, d_theta_2])
    cur = np.matmul(params['A'], pre) + np.matmul(params['B'], u)

    return cur

def discretize_state_space(params):
    # Create edges for each state dimension
    theta_LRs = np.linspace(-params['LR_range'], params['LR_range'], params['bins'])
    theta_1s = np.linspace(-params['12_range'], params['12_range'], params['bins'])
    theta_2s = np.linspace(-params['12_range'], params['12_range'], params['bins'])
    
    dtheta_LRs = np.linspace(-params['u_LR_range'], params['u_LR_range'], params['bins'])
    dtheta_1s = np.linspace(-params['u_LR_range'], params['u_LR_range'], params['bins'])
    dtheta_2s = np.linspace(-params['u_LR_range'], params['u_LR_range'], params['bins'])

    # Create state grid using meshgrid
    grid = np.meshgrid(
        theta_LRs, theta_1s, theta_2s,
        dtheta_LRs, dtheta_1s, dtheta_2s,
        indexing='ij'
    )
    
    # Store edge values
    state_grid = [
        theta_LRs, theta_1s, theta_2s,
        dtheta_LRs, dtheta_1s, dtheta_2s
    ]
    
    # Combine all states into a matrix (n_states × 12)
    states = np.vstack([g.ravel() for g in grid]).T
    
    return states, state_grid

def define_actions(params):
    return np.arange(-params['u_LR_range'], params['u_LR_range']+1, params['u_sample_rate'])

def find_nearest_state(s, state_grid):
    """Find index of nearest discrete state"""
    theta_LR_idx = np.argmin(np.abs(state_grid[0] - s[0]))
    theta_1_idx = np.argmin(np.abs(state_grid[1] - s[1]))
    theta_2_idx = np.argmin(np.abs(state_grid[2] - s[2]))
    dtheta_LR_idx = np.argmin(np.abs(state_grid[3] - s[3]))
    dtheta_1_idx = np.argmin(np.abs(state_grid[4] - s[4]))
    dtheta_2_idx = np.argmin(np.abs(state_grid[5] - s[5]))

    
    # Calculate linear index
    sz = (len(state_grid[0]), len(state_grid[1]), len(state_grid[2]), len(state_grid[3]), 
          len(state_grid[4]), len(state_grid[5]), len(state_grid[6]))
    idx = np.ravel_multi_index((theta_LR_idx, theta_1_idx, theta_2_idx, 
                                dtheta_LR_idx, dtheta_1_idx, dtheta_2_idx), sz)
    
    return idx


def create_environment(params):
    """Create simulation environment"""
    env = {
        'params': d,
        'dynamics': lambda s, a: cartpole_dynamics(s, a, params),
        'reset': lambda: np.array([0, 0, 0.1, 0])  # [x, dx, theta, dtheta]
    }
    return env

# def discretize_state_space(params):
#     """Discretize the state space"""
#     x_edges = np.linspace(-params['x_threshold'], params['x_threshold'], params['x_bins'])
#     dx_edges = np.linspace(-5, 5, params['dx_bins'])
#     theta_edges = np.linspace(-params['theta_threshold'], params['theta_threshold'], params['theta_bins'])
#     dtheta_edges = np.linspace(-5, 5, params['dtheta_bins'])
    
#     # Create state grid
#     state_grid = [x_edges, dx_edges, theta_edges, dtheta_edges]
    
#     # Generate all possible state combinations
#     states = np.array(list(product(x_edges, dx_edges, theta_edges, dtheta_edges))).T
    
#     return states, state_grid

# def define_actions(params):
#     """Define discrete action space"""
#     return np.linspace(-params['max_force'], params['max_force'], 3)  # [-F, 0, +F]

def build_transition_matrix(states, actions, state_grid, params):
    """Build state transition probability matrix P(s'|s,a)"""
    num_states = states.shape[0]
    num_actions = len(actions)
    P = np.zeros((num_states, num_actions, num_states))
    
    for s in range(num_states):
        current_state = states[s, :]
        for a in range(num_actions):
            # Apply dynamics to get next state
            next_state = get_next_state(current_state, actions[a], params)
            
            # Find nearest discrete state
            s_prime = find_nearest_state(next_state, state_grid)
            
            # Update transition probability (deterministic)
            P[s, a, s_prime] = 1
            
    return P

def define_reward_function(states, actions, params):
    """Define reward function R(s,a)"""
    num_states = states.shape[0]
    num_actions = len(actions)
    R = np.zeros((num_states, num_actions))
    
    for s in range(num_states):
        theta_LR, theta_1, theta_2, dtheta_LR, dtheta_1, d_theta_2 = states[s, :]
        
        for a in range(num_actions):
            # Main penalty for angle deviation, secondary for position and control force
            R[s, a] = -(100*np.abs(theta_1) + 100*np.abs(theta_2) + 2*np.abs(theta_LR))
            
            # # Large negative reward for terminal states
            # if abs(x) > params['x_threshold'] or abs(theta) > params['theta_threshold']:
            #     R[s, a] = -100
                
    return R

def value_iteration(states, actions, P, R, gamma, max_iter, tol):
    """Value iteration algorithm"""
    num_states = states.shape[0]
    V = np.zeros(num_states)
    
    for iter in range(max_iter):
        V_prev = V.copy()
        
        for s in range(num_states):
            Q = np.zeros(len(actions))
            for a in range(len(actions)):
                Q[a] = R[s, a] + gamma * np.dot(P[s, a, :], V_prev)
            V[s] = np.max(Q)
        
        # Check convergence
        if np.max(np.abs(V - V_prev)) < tol:
            print(f'Value iteration converged after {iter+1} iterations')
            break

        print(f'theta_LR:{states[s][0]}, theta_1:{states[s][1]}, theta_2:{states[s][2]}, dtheta_LR:{states[s][3]}, dtheta_1:{states[s][4]}, d_theta_2:{states[s][5]}')
            
    # Extract optimal policy
    policy = extract_policy(V, states, actions, P, R, gamma)
    
    return V, policy

def policy_iteration(states, actions, P, R, gamma, max_iter, tol):
    """Policy iteration algorithm"""
    num_states = states.shape[0]
    policy = np.random.randint(len(actions), size=num_states)
    V = np.zeros(num_states)
    
    for iter in range(max_iter):
        # Policy evaluation
        V = evaluate_policy(policy, states, P, R, gamma, tol)
        
        # Policy improvement
        policy_prev = policy.copy()
        policy = extract_policy(V, states, actions, P, R, gamma)
        
        # Check policy stability
        if np.all(policy == policy_prev):
            print(f'Policy iteration converged after {iter+1} iterations')
            break
            
    return V, policy

def evaluate_policy(policy, states, P, R, gamma, tol):
    """Evaluate a given policy"""
    num_states = states.shape[0]
    V = np.zeros(num_states)
    
    for _ in range(1000):  # Inner iteration limit
        V_prev = V.copy()
        
        for s in range(num_states):
            a = policy[s]
            V[s] = R[s, a] + gamma * np.dot(P[s, a, :], V_prev)
            
        # Check convergence
        if np.max(np.abs(V - V_prev)) < tol:
            break
            
    return V

def extract_policy(V, states, actions, P, R, gamma):
    """Extract greedy policy from value function"""
    num_states = states.shape[1]
    policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        Q = np.zeros(len(actions))
        for a in range(len(actions)):
            Q[a] = R[s, a] + gamma * np.dot(P[s, a, :], V)
        policy[s] = np.argmax(Q)
        
    return policy

def test_policy(policy, env, params, state_grid, actions):
    """Test the trained policy"""
    state = [0, 0, 0, 0, 0, 0]
    max_steps = 1000
    history = np.zeros((max_steps, 5))  # [theta_LR, theta_1, theta_2, action, reward]
    
    for t in range(max_steps):
        # Find nearest discrete state index
        s = find_nearest_state(state, state_grid)
        
        # Select action according to policy
        action = actions[policy[s]]
        
        # Apply action
        next_state = get_next_state(state, action, params)
        
        # Calculate reward
        reward = -(100*theta_1 + 100*theta_2 + 2*theta_LR)
        
        # Store history
        history[t] = np.concatenate([state[:3], [action, reward]])
        
        # Check termination conditions
        # if (abs(next_state[0]) > params['x_threshold'] or 
        #     abs(next_state[2]) > params['theta_threshold']):
        #     print(f'Test terminated at step {t+1}')
        #     break
            
        # Update state
        state = next_state
        
    # Plot test results
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(history[0, :t+1])
    plt.title('Cart Position')
    plt.ylabel('x (m)')
    
    plt.subplot(3, 1, 2)
    plt.plot(history[2, :t+1])
    plt.title('Pole Angle')
    plt.ylabel('theta (rad)')
    
    plt.subplot(3, 1, 3)
    plt.plot(history[4, :t+1])
    plt.title('Control Force')
    plt.ylabel('F (N)')
    plt.xlabel('Time step')
    
    plt.tight_layout()
    plt.show()

def plot_results(V, policy, states):
    """Visualize value function and policy"""
    plt.figure(figsize=(12, 5))
    
    # Fixed dx=0, dtheta=0 for visualization
    fixed_dx = 0
    fixed_dtheta = 0
    
    # Find states where dx≈0 and dtheta≈0
    mask = (np.abs(states[1, :] - fixed_dx) < 0.1) & (np.abs(states[3, :] - fixed_dtheta) < 0.1)
    selected_states = states[:, mask]
    selected_V = V[mask]
    selected_policy = policy[mask]
    
    # Extract x and theta values
    x = selected_states[0, :]
    theta = selected_states[2, :]
    
    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(theta.min(), theta.max(), 50)
    Xq, Yq = np.meshgrid(xi, yi)
    
    # Interpolate value function
    Vq = griddata((x, theta), selected_V, (Xq, Yq), method='linear')
    
    # Plot value function
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Xq, Yq, Vq, cmap='viridis')
    ax1.set_title('Value Function (dx=0, dtheta=0)')
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Angle theta')
    ax1.set_zlabel('Value V')
    
    # Interpolate policy
    Pq = griddata((x, theta), selected_policy, (Xq, Yq), method='nearest')
    
    # Plot policy
    plt.subplot(1, 2, 2)
    plt.contourf(Xq, Yq, Pq, levels=np.unique(selected_policy))
    plt.title('Policy (dx=0, dtheta=0)')
    plt.xlabel('Position x')
    plt.ylabel('Angle theta')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def cartpole_dynamics(s, a, params):
    """Second-order cart-pole dynamics model"""
    x, dx, theta, dtheta = s
    F = a
    
    # System parameters
    mc = params['m_cart']
    mp = params['m_pole']
    l = params['l']
    g = params['g']
    dt = params['dt']
    
    # Calculate dynamics
    total_mass = mc + mp
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Pole moment of inertia
    I = (1/3) * mp * l**2
    
    # Calculate accelerations
    temp = (F + mp*l*dtheta**2*sin_theta) / total_mass
    theta_acc = (mp*g*l*sin_theta - mp*l*cos_theta*temp) / (I + mp*l**2)
    x_acc = temp - (mp*l*theta_acc*cos_theta)/total_mass
    
    # Euler integration
    dx_next = dx + x_acc * dt
    x_next = x + dx * dt + 0.5 * x_acc * dt**2
    
    dtheta_next = dtheta + theta_acc * dt
    theta_next = theta + dtheta * dt + 0.5 * theta_acc * dt**2
    
    # Ensure angle is in [-pi, pi]
    theta_next = wrap_to_pi(theta_next)
    
    # Return next state
    return np.array([x_next, dx_next, theta_next, dtheta_next])

def find_nearest_state(s, state_grid):
    """Find index of nearest discrete state"""
    x_idx = np.argmin(np.abs(state_grid[0] - s[0]))
    dx_idx = np.argmin(np.abs(state_grid[1] - s[1]))
    theta_idx = np.argmin(np.abs(state_grid[2] - s[2]))
    dtheta_idx = np.argmin(np.abs(state_grid[3] - s[3]))
    
    # Calculate linear index
    sz = (len(state_grid[0]), len(state_grid[1]), 
          len(state_grid[2]), len(state_grid[3]))
    idx = np.ravel_multi_index((x_idx, dx_idx, theta_idx, dtheta_idx), sz)
    
    return idx

def wrap_to_pi(angle):
    """Map angle to [-pi, pi] range"""
    return (angle + np.pi) % (2*np.pi) - np.pi

def main():
    """Main program"""
    # 1. Initialize parameters
    params = init_parameters()
    
    # 2. Create simulation environment
    # env = create_environment(params)
    
    # 3. Discretize state space (MDP modeling)
    states, state_grid = discretize_state_space(params)
    
    # 4. Define action space
    actions = define_actions(params)
    
    # 5. Build state transition probability matrix P(s'|s,a)
    P = build_transition_matrix(states, actions, state_grid, params)
    
    # 6. Define reward function R(s,a)
    R = define_reward_function(states, actions, params)
    
    # 7. Train RL algorithms
    gamma = 0.95  # Discount factor
    max_iter = 1000
    tol = 1e-4
    
    # 7.1 Value iteration
    V_vi, policy_vi = value_iteration(states, actions, P, R, gamma, max_iter, tol)
    
    # 7.2 Policy iteration
    V_pi, policy_pi = policy_iteration(states, actions, P, R, gamma, max_iter, tol)
    
    # 8. Test policy
    test_policy(policy_pi, env, params, state_grid, actions)
    
    # 9. Visualize results
    plot_results(V_pi, policy_pi, states)

if __name__ == "__main__":
    main()