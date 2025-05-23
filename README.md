# PyTorch版SAC-Auto强化学习算法与应用示例

## 零.SAC-Auto算法:

###### 自定义程度高的SAC-Auto算法，支持部署策略模型、备份训练过程等功能

论文：《Soft Actor-Critic Algorithms and Applications （arXiv: 1812) 》# 不是1801版

| 算法构成     | 说明                 |
| ------------ | -------------------- |
| rl_typing.py | 强化学习数据类型声明 |
| sac_agent.py | SAC-Auto算法         |

### (0).SAC_Agent模块

###### SAC-Auto算法主模块

##### 0.初始化接口

```python
agent = SAC_Agent(env, kwargs=...)      # 初始化算法, 并设置SAC的训练参数
agent.set_buffer(buffer)                # 为算法自定义replay buffer
agent.set_nn(actor, critic, kwargs=...) # 为算法自定义神经网络
# 更多具体接口信息通过help函数查看DocString
```

##### 1.Torch接口

```python
agent.to('cpu') # 将算法转移到指定设备上
agent.cuda(0)   # 将算法转移到cuda0上运算
agent.cpu()     # 将算法转移到cpu上运算
```

##### 2.IO接口

```python
agent.save('./训练备份')              # 存储算法训练过程checkpoint
agent.load('./训练备份')              # 加载算法训练过程checkpoint
agent.export('策略.onnx', kwargs=...) # 部署训练好的onnx策略模型
```

##### 3.训练交互接口

```python
act_array = agent.select_action(obs, kwargs=...) # 环境交互, 基于策略选择-1~1的随机/确定动作
act_array = agent.random_action()                # 环境随机探索, 完全随机产生-1~1的动作
agent.store_memory(transition, kwargs=...)       # 存储环境转移元组(s, a, r, s_, done)
info_dict = agent.learn(kwargs=...)              # 进行一次SAC优化, 返回Loss/Q函数/...
```

##### 4.其余接口/属性 (非用户调用接口，可在派生SAC_Agent模块中覆写)

```python
obs_tensor = agent.state_to_tensor(obs, kwargs=...) # 将Gym返回的1个obs转换成batch_obs, 用于处理混合输入情况, 默认跟随buffer设置
batch_dict = agent.replay_memory(batch_size, kwargs=...) # 经验回放, 用于实现花样经验回放, 默认跟随buffer设置
agent.buffer_len # 算法属性, 查看当前经验个数, 默认跟随buffer设置
agent.use_per # 算法属性, 查看是否使用PER, 默认跟随buffer设置
```

### (1).SAC_Actor模块和SAC_Critic模块

###### 实现自定义 **观测Encoder** + **策略函数** + **Q函数**

##### 0.自定义神经网络要求

- 要求 **观测Encoder** 输入为观测 *batch_obs* 张量，输出形状为(batch, feature_dim)的特征 *batch_feature* 张量。要求forward函数只接受一个位置参数obs，混合观测要求传入的obs为张量字典dict[any, Tensor] / 张量列表list[Tensor] / 张量元组tuple[Tensor, ...]。
- 要求 **策略函数** 输入为特征 *batch_feature* 张量，输出形状为(batch, action_dim)的未经tanh激活的均值 *batch_mu* 张量和对数标准差 *batch_logstd* 张量。要求forward函数只接受一个位置参数feature，形状为(batch, feature_dim)。
- 要求 **Q函数** 输入为特征 *batch_feature* 张量+动作 *batch_action* 张量，输出形状为(batch, 1)的Q值 *batch_q* 张量。要求forward函数只接受一个位置参数 *feature_and_action*，形状为(batch, feature_dim+action_dim)。

##### 1.自定义神经网络示例

```python
FEATURE_DIM = 128
ACTION_DIM = 3

# 自定义观测Encoder
class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ... # user encoder: CNN、RNN、Transformer、GNN ... 
        self.mlp = nn.Sequential(
            nn.Linear(..., FEATURE_DIM),
            nn.ReLU(True),
        )
    def forward(self, observation):
        feature = self.mlp(self.encoder(observation))
        return feature

encoder_net = MyEncoder()

# 自定义策略函数
class MyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
	self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.ReLU(True),
	    nn.Linear(128, ACTION_DIM), # no activation
        )
    def forward(self, feature):
        return self.mlp(feature)

mu_net, logstd_net = MyPolicy(), MyPolicy()

# 自定义TwinQ函数
class MyQfun(nn.Module):
    def __init__(self):
        super().__init__()
	self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM + ACTION_DIM, 128),
            nn.ReLU(True),
	    nn.Linear(128, 1), # no activation
        )
    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

q1_net, q2_net = MyQfun(), MyQfun()

# 为算法设置神经网络
actor = SAC_Actor(encoder_net, mu_net, logstd_net, kwargs=...) # 实例化actor网络
critic = SAC_Critic(encoder_net, q1_net, q2_net)               # 实例化critic网络

agent.set_nn(
    actor, 
    critic, 
    actor_optim_cls = th.optim.Adam, 
    critic_optim_cls = th.optim.Adam, 
    copy = True
)
```
