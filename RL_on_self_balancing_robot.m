%% 二阶平衡小车强化学习控制 - 主程序
clear; clc; close all;

%% 1. 参数初始化
params = init_parameters();

%% 2. 创建仿真环境
env = create_environment(params);

%% 3. 离散化状态空间（MDP建模）
[states, state_grid] = discretize_state_space(params);

%% 4. 定义动作空间
actions = define_actions(params);

%% 5. 构建状态转移概率矩阵P(s'|s,a)
P = build_transition_matrix(states, actions, env, state_grid);

%% 6. 定义奖励函数R(s,a)
R = define_reward_function(states, actions, params);

%% 7. 训练强化学习算法
gamma = 0.95;  % 折扣因子
max_iter = 1000;
tol = 1e-4;

% 7.1 值迭代算法
[V_vi, policy_vi] = value_iteration(states, actions, P, R, gamma, max_iter, tol);

% 7.2 策略迭代算法
[V_pi, policy_pi] = policy_iteration(states, actions, P, R, gamma, max_iter, tol);

%% 8. 仿真测试
test_policy(policy_pi, env, params, state_grid, actions);

%% 9. 结果可视化
plot_results(V_pi, policy_pi, states);

%% ========== 函数定义 ========== %%

function params = init_parameters()
    % 物理参数
    params.m_cart = 1.0;     % 小车质量 (kg)
    params.m_pole = 0.1;     % 摆杆质量 (kg)
    params.l = 0.5;          % 摆杆长度 (m)
    params.g = 9.81;         % 重力加速度 (m/s^2)
    params.dt = 0.02;        % 时间步长 (s)
    
    % 强化学习参数
    params.max_force = 10;   % 最大控制力 (N)
    params.x_threshold = 2.4; % 小车位置阈值 (m)
    params.theta_threshold = 12 * pi/180; % 角度阈值 (rad)
    
    % 状态离散化参数
    params.x_bins = 6;      % 位置分箱数
    params.dx_bins = 6;     % 速度分箱数
    params.theta_bins = 6;  % 角度分箱数
    params.dtheta_bins = 6; % 角速度分箱数
end

function env = create_environment(params)
    % 创建仿真环境结构体
    env.params = params;
    
    % 动力学方程
    env.dynamics = @(s, a) cartpole_dynamics(s, a, params);
    
    % 重置环境函数
    env.reset = @() [0; 0; 0.1; 0]; % [x, dx, theta, dtheta]
end

function [states, state_grid] = discretize_state_space(params)
    % 状态空间离散化
    x_edges = linspace(-params.x_threshold, params.x_threshold, params.x_bins);
    dx_edges = linspace(-5, 5, params.dx_bins);
    theta_edges = linspace(-params.theta_threshold, params.theta_threshold, params.theta_bins);
    dtheta_edges = linspace(-5, 5, params.dtheta_bins);
    
    % 创建状态网格
    [X, DX, THETA, DTHETA] = ndgrid(x_edges, dx_edges, theta_edges, dtheta_edges);
    state_grid = {x_edges, dx_edges, theta_edges, dtheta_edges};
    
    % 将所有可能状态组合存储为矩阵
    states = [X(:), DX(:), THETA(:), DTHETA(:)]';
end

function actions = define_actions(params)
    % 定义离散动作空间
    actions = linspace(-params.max_force, params.max_force, 3); % [-F, 0, +F]
end

function P = build_transition_matrix(states, actions, env, state_grid)
    % 初始化转移概率矩阵
    num_states = size(states, 2);
    num_actions = numel(actions);
    P = zeros(num_states, num_actions, num_states);
    
    % 构建转移概率矩阵
    for s = 1:num_states
        current_state = states(:, s)';
        for a = 1:num_actions
            % 应用动力学模型得到下一状态
            next_state = env.dynamics(current_state, actions(a));
            
            % 找到最近的离散状态
            s_prime = find_nearest_state(next_state, state_grid);
            
            % 更新转移概率
            P(s, a, s_prime) = 1; % 确定性转移
        end
    end
end

function R = define_reward_function(states, actions, params)
    % 初始化奖励矩阵
    num_states = size(states, 2);
    num_actions = numel(actions);
    R = zeros(num_states, num_actions);
    
    % 定义奖励函数
    for s = 1:num_states
        x = states(1, s);
        theta = states(3, s);
        
        for a = 1:num_actions
            % 主要惩罚角度偏差，次要惩罚位置偏移和控制力
            R(s, a) = - (theta^2 + 0.1*x^2 + 0.001*actions(a)^2);
            
            % 如果超出阈值，给予大的负奖励（终止状态）
            if abs(x) > params.x_threshold || abs(theta) > params.theta_threshold
                R(s, a) = -100;
            end
        end
    end
end

function [V, policy] = value_iteration(states, actions, P, R, gamma, max_iter, tol)
    % 初始化值函数
    V = zeros(size(states, 2), 1);
    
    for iter = 1:max_iter
        V_prev = V;
        
        for s = 1:size(states, 2)
            Q = zeros(numel(actions), 1);
            for a = 1:numel(actions)
                Q(a) = R(s, a) + gamma * squeeze(P(s, a, :))' * V_prev;
            end
            V(s) = max(Q);
        end
        
        % 检查收敛
        if max(abs(V - V_prev)) < tol
            fprintf('值迭代在 %d 次迭代后收敛\n', iter);
            break;
        end
    end
    
    % 提取最优策略
    policy = extract_policy(V, states, actions, P, R, gamma);
end

function [V, policy] = policy_iteration(states, actions, P, R, gamma, max_iter, tol)
    % 初始化随机策略
    policy = randi(numel(actions), size(states, 2), 1);
    V = zeros(size(states, 2), 1);
    
    for iter = 1:max_iter
        % 策略评估
        V = evaluate_policy(policy, states, P, R, gamma, tol);
        
        % 策略改进
        policy_prev = policy;
        policy = extract_policy(V, states, actions, P, R, gamma);
        
        % 检查策略是否稳定
        if all(policy == policy_prev)
            fprintf('策略迭代在 %d 次迭代后收敛\n', iter);
            break;
        end
    end
end

function V = evaluate_policy(policy, states, P, R, gamma, tol)
    % 初始化值函数
    V = zeros(size(states, 2), 1);
    
    for iter = 1:1000  % 内部迭代限制
        V_prev = V;
        
        for s = 1:size(states, 2)
            a = policy(s);
            V(s) = R(s, a) + gamma * squeeze(P(s, a, :))' * V_prev;
        end
        
        % 检查收敛
        if max(abs(V - V_prev)) < tol
            break;
        end
    end
end

function policy = extract_policy(V, states, actions, P, R, gamma)
    % 从值函数中提取贪心策略
    policy = zeros(size(states, 2), 1);
    
    for s = 1:size(states, 2)
        Q = zeros(numel(actions), 1);
        for a = 1:numel(actions)
            Q(a) = R(s, a) + gamma * squeeze(P(s, a, :))' * V;
        end
        [~, policy(s)] = max(Q);
    end
end

function test_policy(policy, env, params, state_grid, actions)
    % 测试训练好的策略
    state = env.reset();
    max_steps = 1000;
    history = zeros(6, max_steps); % [x; dx; theta; dtheta; action; reward]
    
    for t = 1:max_steps
        % 找到当前状态对应的离散状态索引
        s = find_nearest_state(state, state_grid);
        
        % 根据策略选择动作
        action = actions(policy(s));
        
        % 应用动作
        next_state = env.dynamics(state, action);
        
        % 计算奖励
        reward = - (next_state(3)^2 + 0.1*next_state(1)^2 + 0.001*action^2);
        
        % 存储历史
        history(:, t) = [state; action; reward];
        
        % 检查终止条件
        if abs(next_state(1)) > params.x_threshold || ...
           abs(next_state(3)) > params.theta_threshold
            fprintf('测试在第 %d 步终止\n', t);
            break;
        end
        
        % 更新状态
        state = next_state;
    end
    
    % 绘制测试结果
    figure;
    subplot(3,1,1);
    plot(history(1,1:t));
    title('小车位置');
    ylabel('x (m)');
    
    subplot(3,1,2);
    plot(history(3,1:t));
    title('摆杆角度');
    ylabel('theta (rad)');
    
    subplot(3,1,3);
    plot(history(5,1:t));
    title('控制力');
    ylabel('F (N)');
    xlabel('时间步');
end

function plot_results(V, policy, states)
    % 可视化值函数和策略
    figure;
    
    % 固定dx=0, dtheta=0，可视化x和theta平面的值函数
    fixed_dx = 0;
    fixed_dtheta = 0;
    
    % 找到dx≈0和dtheta≈0的状态索引
    mask = (abs(states(2,:) - fixed_dx) < 0.1) & (abs(states(4,:) - fixed_dtheta) < 0.1);
    selected_states = states(:, mask);
    selected_V = V(mask);
    
    % 提取x和theta值
    x = selected_states(1,:);
    theta = selected_states(3,:);
    
    % 创建网格数据
    [Xq, Yq] = meshgrid(linspace(min(x), max(x), 50), linspace(min(theta), max(theta), 50));
    Vq = griddata(x, theta, selected_V, Xq, Yq);
    
    % 绘制值函数
    subplot(1,2,1);
    surf(Xq, Yq, Vq);
    title('值函数 (dx=0, dtheta=0)');
    xlabel('位置 x');
    ylabel('角度 theta');
    zlabel('值 V');
    
    % 绘制策略
    selected_policy = policy(mask);
    Pq = griddata(x, theta, selected_policy, Xq, Yq);
    
    subplot(1,2,2);
    contourf(Xq, Yq, Pq);
    title('策略 (dx=0, dtheta=0)');
    xlabel('位置 x');
    ylabel('角度 theta');
    colorbar;
end

function s_next = cartpole_dynamics(s, a, params)
    % 二阶平衡小车动力学模型
    x = s(1); dx = s(2);
    theta = s(3); dtheta = s(4);
    F = a;
    
    % 系统参数
    mc = params.m_cart;
    mp = params.m_pole;
    l = params.l;
    g = params.g;
    dt = params.dt;
    
    % 计算动力学
    total_mass = mc + mp;
    sin_theta = sin(theta);
    cos_theta = cos(theta);
    
    % 摆杆的转动惯量
    I = (1/3) * mp * l^2;
    
    % 计算加速度
    temp = (F + mp*l*dtheta^2*sin_theta) / total_mass;
    theta_acc = (mp*g*l*sin_theta - mp*l*cos_theta*temp) / (I + mp*l^2);
    x_acc = temp - (mp*l*theta_acc*cos_theta)/total_mass;
    
    % 欧拉积分
    dx_next = dx + x_acc * dt;
    x_next = x + dx * dt + 0.5 * x_acc * dt^2;
    
    dtheta_next = dtheta + theta_acc * dt;
    theta_next = theta + dtheta * dt + 0.5 * theta_acc * dt^2;
    
    % 确保角度在[-pi, pi]范围内
    theta_next = wrapToPi(theta_next);
    
    % 返回下一状态
    s_next = [x_next; dx_next; theta_next; dtheta_next];
end

function idx = find_nearest_state(s, state_grid)
    % 找到最接近的离散状态索引
    [~, x_idx] = min(abs(state_grid{1} - s(1)));
    [~, dx_idx] = min(abs(state_grid{2} - s(2)));
    [~, theta_idx] = min(abs(state_grid{3} - s(3)));
    [~, dtheta_idx] = min(abs(state_grid{4} - s(4)));
    
    % 计算线性索引
    sz = [length(state_grid{1}), length(state_grid{2}), ...
          length(state_grid{3}), length(state_grid{4})];
    idx = sub2ind(sz, x_idx, dx_idx, theta_idx, dtheta_idx);
end

function angle = wrapToPi(angle)
    % 将角度映射到[-pi, pi]范围
    angle = mod(angle + pi, 2*pi) - pi;
end