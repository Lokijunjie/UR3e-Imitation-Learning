# -*- coding: utf-8 -*-
"""
策略训练示例
 Created on Sat Nov 04 2023 15:37:28
 Modified on 2025-5-20 15:37:28
 
 @auther: Junjie Wang
"""
#


'''算法定义'''
import numpy as np
import torch as th
import torch.nn as nn
from copy import deepcopy
from sac_agent import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./runs/experiment_4')  # 设置日志文件保存路径


# 1.定义经验回放（取决于观测和动作数据结构）
class Buffer(BaseBuffer):
    def __init__(self, memory_size, obs_space, act_space):
        super(Buffer, self).__init__()
        # 数据类型表示
        self.device = 'cuda'
        self.obs_space = obs_space
        self.act_space = act_space
        # buffer属性
        self._ptr = 0    # 当前位置
        self._idxs = [0] # PER记住上一次采样位置, 一维list或ndarray
        self._memory_size = int(memory_size) # 总容量
        self._current_size = 0               # 当前容量
        # buffer容器
        obs_shape = obs_space.shape or (1, )
        act_shape = act_space.shape or (1, ) # NOTE DiscreteSpace的shape为(), 设置collections应为(1, )
        self._data = {}
        self._data["s"] = np.empty((memory_size, *obs_shape), dtype=obs_space.dtype) # (size, *obs_shape, )连续 (size, 1)离散
        self._data["s_"] = deepcopy(self._data["s"])                                 # (size, *obs_shape, )连续 (size, 1)离散
        self._data["a"] = np.empty((memory_size, *act_shape), dtype=act_space.dtype) # (size, *act_shape, )连续 (size, 1)离散
        self._data["r"] = np.empty((memory_size, 1), dtype=np.float32)               # (size, 1)
        self._data["done"] = np.empty((memory_size, 1), dtype=bool)                  # (size, 1) 

    def reset(self, *args, **kwargs):
        self._ptr = 0
        self._idxs = [0]
        self._current_size = 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._data.values())

    def push(self, transition, terminal=None, **kwargs):
        self._data["s"][self._ptr] = transition[0]
        self._data["a"][self._ptr] = transition[1]
        self._data["r"][self._ptr] = transition[2]
        self._data["s_"][self._ptr] = transition[3]
        self._data["done"][self._ptr] = transition[4]
        # update
        self._ptr = (self._ptr + 1) % self._memory_size                     # 更新指针
        self._current_size = min(self._current_size + 1, self._memory_size) # 更新容量

    def __len__(self):
        return self._current_size 
    
    def sample(self, batch_size=1, *, idxs=None, rate=None, **kwargs):
        self._idxs = idxs or np.random.choice(self._current_size, size=batch_size, replace=False)
        batch = {k: th.FloatTensor(self._data[k][self._idxs]).to(self.device) for k in self._data.keys()}
        return batch
    
    def state_to_tensor(self, state, use_rnn=False):
        return th.FloatTensor(state).unsqueeze(0).to(self.device)
    
# 2.定义神经网络（取决于观测数据结构）
# Q网络
QEncoderNet = nn.Identity

class QNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(QNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim+act_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

# Pi网络
class PiEncoderNet(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super(PiEncoderNet, self).__init__()
        obs_dim = np.prod(obs_shape)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, feature_dim),
            nn.ReLU(True),
        )
    def forward(self, obs):
        return self.mlp(obs)
    
class PiNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PiNet, self).__init__()
        self.mlp = nn.Linear(feature_dim, act_dim)
    def forward(self, feature):
        return self.mlp(feature)



'''实例化环境'''
from DMP_SAC_Env import  NormalizedActionsWrapper, dmpEnv
env = NormalizedActionsWrapper(dmpEnv())
obs_space = env.observation_space
act_space = env.action_space



'''实例化算法'''
# 1.缓存设置
buffer = Buffer(10000, obs_space, act_space)


encoder_layer = PiEncoderNet(obs_space.shape, 128)
# 打印Encoder网络shape
print("Encoder网络输出shape: ", encoder_layer(th.zeros(1, *obs_space.shape)).shape)

# 2.神经网络设置
actor = SAC_Actor(
        PiEncoderNet(obs_space.shape, 128),
        PiNet(128, act_space.shape[0]),
        PiNet(128, act_space.shape[0]),
    )
critic = SAC_Critic(
        QEncoderNet(),
        QNet(obs_space.shape[0], act_space.shape[0]),
        QNet(obs_space.shape[0], act_space.shape[0]),
    )

# 3.算法设置
agent = SAC_Agent(env,lr_actor=2e-3,lr_critic=2e-3)
agent.set_buffer(buffer)
agent.set_nn(actor, critic)
agent.cuda()


final_obs = env.reset()
rl_w=env.w.copy()
'''训练LOOP''' 
MAX_EPISODE = 200
for episode in range(MAX_EPISODE):
    ep_reward = 0
    obs = env.reset()

    for steps in range(env.max_episode_steps):
        # 决策
        act = agent.select_action(obs)
        # 仿真
        next_obs, reward, done, info = env.step(act)
        ep_reward += reward
        
        # 缓存
        agent.store_memory((obs, act, reward, next_obs, done))
        
        # 优化
        loss_dict = agent.learn()

        # 记录标量：奖励和损失等,包括平均奖励
        writer.add_scalar('Reward/Episode', ep_reward, episode)
        writer.add_scalar('Reward/Step', reward, steps)
        writer.add_scalar('Reward/Mean', ep_reward / (steps + 1), steps )

        # 记录损失
        if loss_dict['q_loss'] is not None:
            writer.add_scalar('Loss/Q-Critic', loss_dict['q_loss'], episode * env.max_episode_steps + steps)
        if loss_dict['actor_loss'] is not None:
            writer.add_scalar('Loss/Actor', loss_dict['actor_loss'], episode * env.max_episode_steps + steps)
        if loss_dict['alpha_loss'] is not None:
            writer.add_scalar('Loss/Alpha', loss_dict['alpha_loss'], episode * env.max_episode_steps + steps)

        # 回合结束
        if info["terminal"]:
            final_obs = deepcopy(next_obs)
            rl_w = deepcopy(act)
            mean_reward = ep_reward / (steps + 1)
            print('回合: ', episode, '| 累积奖励: ', round(ep_reward, 2), '| 平均奖励: ', round(mean_reward, 2), '| 状态: ', info, '| 步数: ', steps)
            break
        else:
            obs = deepcopy(next_obs)
    # break loops
    if done:
        break

writer.close() # 关闭tensorboard
ori个inal_shaee = env.q_demo.shape
show_obs = np.zeros(ori个inal_shaee)
# 将final_obs切片为3个部分
# show_obs[:,0] = final_obs[0:75]
# show_obs[:,1] = final_obs[75:150]
show_obs = final_obs.reshape(ori个inal_shaee)
# 绘制obs的YZ平面轨迹
import matplotlib.pyplot as plt

# 绘制obs的YZ平面轨迹
plt.figure(figsize=(6, 6))
plt.plot(show_obs[:, 0], show_obs[:, 1], label="Learned Trajectory", color="C0")
plt.plot(env.q_demo[:, 0], env.q_demo[:, 1], label="Demonstration Trajectory", color="red")
plt.xlabel("Y-axis")
plt.ylabel("Z-axis")
plt.title("Trajectory in YZ Plane")
plt.legend()
plt.grid()
plt.show()

# 保存show_obs
import pandas as pd

df = pd.DataFrame(show_obs, columns=["Y", "Z"])  # 为数据添加列名
df.to_csv("RL—letter4_trajectory.csv", index=False)  # 保存为CSV文件


# traj_error = np.linalg.norm(final_obs - env.q_demo)
# print("trajectory error",traj_error)

# from path_plan_env.dmp import DynamicMovementPrimitive
# from path_plan_env.utils import smooth_trajectory, vel, normalize_vector
# dmp = DynamicMovementPrimitive(_a=15, _ng=100, _stb=False)
# data = load_demo('/home/wangjunjie/pyrdmp/examples/data/ur3e_tip_positions_Smax.txt')


    #end for
#end for
agent.export("./path_plan_env/policy_static.onnx") # 导出策略模型
# agent.save("./checkpoint") # 存储算法训练进度
# agent.load("./checkpoint") # 加载算法训练进度







r'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    佛祖保佑       永无BUG
'''
