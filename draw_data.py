from matplotlib import pyplot as plt
import pandas as pds

# 第一幅图片：训练的平均长度
fig, ax1 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
len_mean = pds.read_csv("run-PPO_2-tag-rollout_ep_len_mean.csv")
ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="#3399FF")
#ax1.set_xticks(np.arange(0, 24, step=2))
ax1.set_xlabel("timesteps")
ax1.set_ylabel("Average Episode Length(steps)", color="#3399FF")
ax1.set_title("Average Episode Length")
plt.show()
fig.savefig(fname='./figures/ep_len_mean'+'.pdf', format='pdf')