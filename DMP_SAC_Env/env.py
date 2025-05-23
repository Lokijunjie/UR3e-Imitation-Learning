# -*- coding: utf-8 -*-
"""
DMP强化学习环境
 Created on Tue May 16 2023 17:54:17
 Modified on 2025-5-20 17:38:00
 
 @auther: Junjie Wang
"""
#
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from copy import deepcopy
from pathlib import Path
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon

__all__ = [ "dmpEnv"]



# 示教轨迹文件路径
demo_path = '/home/wangjunjie/DMP-SAC/data/letter_4_trajectory.csv'

from .dmp import DynamicMovementPrimitive
from .utils import *
class dmpEnv(gym.Env):
    def __init__(self, max_episode_steps=500, old_gym_style=True):
        """
        Args:
            max_episode_steps (int): 最大仿真步数. 默认500.
            old_gym_style (bool): 是否采用老版gym接口. 默认True.
        """
        self.max_episode_steps = max_episode_steps
        self.__need_reset = True
        self.__old_gym = old_gym_style

        self.dmp = DynamicMovementPrimitive(_a=10, _ng=100, _stb=False)
        # 示例轨迹：读取示教轨迹文件
        q = load_demo(demo_path)

        print("q.len: ",q.shape[0])

        # q = parse_demo(data)
        # 间隔0.1s生成t，数量与q相同
        t = np.linspace(0, (q.shape[0] - 1) * 0.1, q.shape[0])

        print("t.shape:", t.shape)
        s = self.dmp.phase(t)
        psv = self.dmp.distributions(s)

        # 归一化时间向量
        t = normalize_vector(t)

        # 计算速度和加速度
        dq, ddq = np.zeros((2, q.shape[0], q.shape[1]))

        for i in range(q.shape[1]):
            dq[:, i] = vel(q[:, i], t)
            ddq[:, i] = vel(dq[:, i], t)
     
         # Imitation Learning
        ftarget = np.zeros(q.shape)
        w = np.zeros((self.dmp.ng, q.shape[0]))

        print('Imitation start')

        print("t length:",len(t))
        print("f_target length:", len(ftarget))
        print("ddx length:", len(ddq))
        print("x length:", len(q))
        print("dx length:", len(dq))

        for i in range(q.shape[1]):
            ftarget[:, i], w[:, i] = self.dmp.imitate(q[:, i], dq[:, i], 
                    ddq[:, i], t, s, psv)

        print('Imitation done')

        # Generate the Learned trajectory
        x, dx, ddx = np.zeros((3, q.shape[0], q.shape[1]))

        for i in range(q.shape[1]):
            ddx[:, i], dx[:, i], x[:, i] = self.dmp.generate(w[:, i], q[0, i], 
                    q[-1, i], t, s, psv)
            
        # 打印轨迹误差 L2范数
        print("L2 Norm Error:", np.linalg.norm(x - q, axis=0))  # 对每个维度计算L2范数

        # 保存轨迹和目标
        self.q_demo = q
        self.dq_demo = dq
        self.ddq_demo = ddq
        self.t_demo = t
        self.w = w  # 非线性强迫项参数
        self.s = s  # 归一化时间向量
        self.psv = psv
        self.x = x


        low_obs = np.array([-0.16, 0.25] * q.shape[0])
        up_obs = np.array([-0.13, 0.295] * q.shape[0])

        self.observation_space = spaces.Box(low_obs, up_obs, dtype=np.float32)
        print("observation_space:", self.observation_space.shape)
        low_action = np.array([-20, -20 ] * self.dmp.ng)
        up_action = np.array([20, 20] * self.dmp.ng)

        self.action_space = spaces.Box(low_action,up_action, dtype=np.float32)

    def reset(self):
        self.__need_reset = False
        self.time_steps = 0
        self.obs = self.x.reshape(-1)
        # _, _, self.obs = self.dmp.generate(self.w, self.q_demo[0], self.q_demo[-1], self.t_demo, self.s, self.psv)
        # obs = np.zeros((self.q_demo.shape[0], self.q_demo.shape[1]), dtype=np.float32)
        
        # # 生成轨迹
        # for i in range(self.q_demo.shape[1]):
        #     _, _, obs[:, i] = self.dmp.generate(self.w[:, i], self.q_demo[0, i], 
        #             self.q_demo[-1, i], self.t_demo, self.s, self.psv)
        # print("obs.shape:", obs.shape)
        # # 将obs变为一维数组
        # self.obs = obs.reshape(-1)

        if self.__old_gym:
            return self.obs
        return self.obs, {}
    
    def step(self, action):
        assert not self.__need_reset, "调用step前必须先reset"
        
        w = action  # 更新DMP的参数W
        
        # 将action由总长度为self.dmp.ng*2的一维数组变成二维数组
        mul_action = np.zeros((self.dmp.ng, self.q_demo.shape[1]))
        mul_action = w.reshape(self.dmp.ng,-1)
        obs = np.zeros((self.q_demo.shape[0], self.q_demo.shape[1]), dtype=np.float32)
        # print("action.shape:", action.shape)
        # print("obs.shape:", self.obs.shape)
        # 使用DMP生成轨迹
        for i in range(self.q_demo.shape[1]):
            _, _, obs[:, i] = self.dmp.generate(mul_action[:, i], self.q_demo[0, i], 
                    self.q_demo[-1, i], self.t_demo, self.s, self.psv)
        #将obs变为一维数组
        obs = obs.reshape(-1)
        # 状态转移（使用DMP生成轨迹）
        obs = np.clip(obs , self.observation_space.low, self.observation_space.high)
        self.time_steps += 1


         # 计算奖励：基于轨迹与示范轨迹的差异
        reward, done, info = self._get_reward(obs)

        # 检查是否结束
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["terminal"] = True
            self.w = w  # 更新DMP的参数W
            self.__need_reset = True
        else:
            info["terminal"] = False

        # self.obs = deepcopy(obs)
        
        if self.__old_gym:
            return self.obs, reward, done, info
        return obs, reward, done, truncated, info

    def _get_reward(self, obs):
        """计算奖励 - 基于轨迹与示范轨迹的差异，并加入越界惩罚、成功奖励和平滑度惩罚"""
        
        q_demo = self.q_demo.reshape(-1)  # 将示范轨迹展平为一维数组

        # 计算轨迹差异
        traj_error = np.linalg.norm(obs - q_demo)  # 使用L2范数
        reward = -100*traj_error  # 奖励是负的轨迹误差（误差小，奖励大）

        # # 计算速度（位置变化量）
        # dt = self.t_demo[1] - self.t_demo[0]  # 时间间隔
        # velocities = np.diff(obs, axis=0) / dt  # 计算相邻位置点的差分，得到速度
        # # 在速度的开头插入一个零作为第一个点的速度
        # velocities = np.insert(velocities, 0, 0, axis=0)
        # # 计算速度的均方差
        # velocity_msd = np.mean(np.square(velocities-self.dq_demo), axis=0)  # 计算速度的均值
        
        
        # # 平滑度惩罚：速度的均方差
        # reward -= 100 * velocity_msd # 惩罚速度波动大的轨迹

        # 约束：越界惩罚
        num_over = 0
        for o, l, u in zip(obs, self.observation_space.low, self.observation_space.high):
            if o <= l or o >= u:
                num_over += 2  # 越界每个点加2的惩罚
        
        reward -= num_over  # 把越界惩罚加到总奖励中

        # 成功奖励：如果轨迹误差小于阈值，并且没有越界，给额外奖励
        done = False
        if traj_error < 0.025 and num_over == 0:  # 轨迹误差很小且没有越界
            reward += 1000  # 成功奖励
            done = True  # 认为任务完成

        info = {
            'traj_error': traj_error,
            'num_over': num_over,
            'done': done
        }

        return reward, done, info


    
    def render(self, mode="human"):
        """可视化当前轨迹"""
        assert not self.__need_reset, "调用render前必须先reset"
        plt.clf()
        plt.axis('equal')
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.plot(self.q_demo, label="示范轨迹")
        plt.plot(self.obs, label="当前轨迹")
        plt.legend(loc='best')
        plt.title("DMP Training Progress")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.grid(True)
        plt.pause(0.001)

    def close(self):
        """关闭可视化"""
        self.__render_not_called = True
        self.__need_reset = True
        plt.close()




#----------------------------- ↓↓↓↓↓ 环境-算法适配 ↓↓↓↓↓ ------------------------------#
class NormalizedActionsWrapper(gym.ActionWrapper):
    """非-1~1的连续动作空间环境装饰器"""
    def __init__(self, env):
        super(NormalizedActionsWrapper, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), '只用于Box动作空间'
  
    # 将神经网络输出转换成gym输入形式
    def action(self, action): 
        # 连续情况 scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)
        action = np.clip(action, lb, ub)
        return action

    # 将gym输入形式转换成神经网络输出形式
    def reverse_action(self, action):
        # 连续情况 normalized action [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return np.clip(action, -1.0, 1.0)
       








if __name__ == '__main__':
    # MAP.show()
    env = DynamicPathPlanning()
    for ep in range(10):
        print(f"episode{ep}: begin")
        obs = env.reset()
        while 1:
            try:
                env.render()
                obs, rew, done, info = env.step(np.array([0.5, 0.2]))
                print(info)
            except AssertionError:
                break
        #env.plot(f"output{ep}")
        print(f"episode{ep}: end")