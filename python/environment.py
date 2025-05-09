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

class DiscreteEnv(gym.Env):
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

        
    def step(self,action):

        dt=1/100
        '''动作的间隔时间'''
        # 参数
        m_1 = 0.9
        m_2 = 0.1
        r = 0.0335
        L_1 = 0.126
        L_2 = 0.390
        l_1 = L_1/2
        l_2 = L_2/2
        g = 9.8
        I_1 = (1/12)*m_1*L_1^2
        I_2 = (1/12)*m_2*L_2^2
        # p矩
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
        p_33 = m_1*l_1^2 + m_2*L_1^2 + I_1
        p_34 = m_2*L_1*l_2
        p_41 = (r/2)*m_2*l_2
        p_42 = (r/2)*m_2*l_2
        p_43 = m_2*L_1*l_2
        p_44 = m_2*l_2^2 + I_2
        p=np.matrix([[p_11,p_12,p_13,p_14],[p_21,p_22,p_23,p_24],[p_31,p_32,p_33,p_34],[p_41,p_42,p_43,p_44]])
        # q矩
        q_11  = 0
        q_12  = 0
        q_13  = 0
        q_14  = 0
        q_15  = 0
        q_16  = 0
        q_17  = 0
        q_18  = 0
        q_19  = 1
        q_110 = 0
        q_21  = 0
        q_22  = 0
        q_23  = 0
        q_24  = 0
        q_25  = 0
        q_26  = 0
        q_27  = 0
        q_28  = 0
        q_29  = 0
        q_210 = 1
        q_31  = 0
        q_32  = 0
        q_33  = (m_1*l_1 + m_2*L_1)*g
        q_34  = 0
        q_35  = 0
        q_36  = 0
        q_37  = 0
        q_38  = 0
        q_39  = 0
        q_310 = 0
        q_41  = 0
        q_42  = 0
        q_43  = 0
        q_44  = m_2*g*l_2
        q_45  = 0
        q_46  = 0
        q_47  = 0
        q_48  = 0
        q_49  = 0
        q_410 = 0
        q=np.matrix([[q_11,q_12,q_13,q_14,q_15,q_16,q_17,q_18,q_19,q_110],[q_21,q_22,q_23,q_24,q_25,q_26,q_27,q_28,q_29,q_210],[q_31,q_32,q_33,q_34,q_35,q_36,q_37,q_38,q_39,q_310],[q_41,q_42,q_43,q_44,q_45,q_46,q_47,q_48,q_49,q_410]])

        # 计算结果
        temp = p.I @q
        A=np.append([[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],temp[:,0:8], axis=0)
        B=np.append([[0,0],[0,0],[0,0],[0,0]],temp[:,8:10], axis=0)
        
        next_state = self.state.copy()
        # 计算下一个状态
        
        



    
        






