import numpy as np
import matplotlib.pyplot as plt

def time_reward(t):
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    k = 0.5  
    t_deadline = 15  
    reward = sigmoid(-k * (t - t_deadline))
    return reward

# 生成时间点
x = np.linspace(0, 30, 1000)  # 从0到30，生成1000个点
y = time_reward(x)

# 创建图形
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'b-', linewidth=3, label='时间奖励函数')

# 标记关键点
t_deadline = 20
plt.axvline(x=t_deadline, color='r', linestyle='--', alpha=0.7, label=f'截止时间 = {t_deadline}')

plt.show()

