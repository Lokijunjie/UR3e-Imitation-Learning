# 基于SAC算法改进的DMP示教学习

## 依赖项
python 3.9.21  
torch  1.13.1+cuda11.6

## 训练
```
python demo_train.py
```

## SAC算法:

###### 自定义程度高的SAC算法，支持部署策略模型、备份训练过程等功能

论文：《Soft Actor-Critic Algorithms and Applications （arXiv: 1812) 》# 不是1801版

| 算法构成     | 说明                 |
| ------------ | -------------------- |
| rl_typing.py | 强化学习数据类型声明 |
| sac_agent.py | SAC算法         |





