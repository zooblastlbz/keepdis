import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 读取状态风险分值的字典
risk_scores = {
    'Distraction Detection': {
        'Normal Driving': 0.0, 'Drinking': 0.5, 'Phoning Left': 0.8, 'Phoning Right': 0.8, 
        'Texting Left': 0.9, 'Texting Right': 0.9, 'Touching Hairs & Makeup': 0.7, 
        'Adjusting Glasses': 0.3, 'Reaching Behind': 0.6, 'Dropping (Fatigue)': 1.0
    },
    'Emotion Recognition': {
        'Angry': 0.8, 'Disgust': 0.7, 'Fear': 0.9, 'Happy': 0.2, 'Sad': 0.6, 'Surprise': 0.5
    },
    'Fatigue Detection': {
        'Non-Drowsy': 0.0, 'Drowsy': 1.0
    }
}

# 任务权重
weights = {
    'Distraction Detection': 0.3,
    'Emotion Recognition': 0.2,
    'Fatigue Detection': 0.5
}

# 滑动窗口长度（秒）
window_length = 30

# 读取CSV文件中的状态数据
data = pd.read_csv('/home/users/ntu/chih0001/scratch/VLM/LLaVA/risk_assessment/driver_states.csv')

# 初始化结果
combined_risk_scores = []

# 遍历所有时间点，模拟滑动窗口
for current_time in range(0, data['timestamp'].max() + 1):
    # 选择滑动窗口内的状态
    window_data = data[(data['timestamp'] >= current_time - window_length) & (data['timestamp'] < current_time)]
    
    # 初始化滑动窗口内的风险分数
    total_risk_score = 0.0

    # 遍历滑动窗口内的状态
    for index, row in window_data.iterrows():
        task = row['task']
        state = row['state']
        duration = row['duration']

        # 计算该状态的风险分数
        task_risk_score = risk_scores[task].get(state, 0)  # 如果状态不存在，则风险分数为0
        weighted_score = (duration / window_length) * weights[task] * task_risk_score
        total_risk_score += weighted_score

    # 当前秒的风险分数
    current_data = data[data['timestamp'] == current_time]
    current_risk = 0.0
    for index, row in current_data.iterrows():
        task = row['task']
        state = row['state']
        task_risk_score = risk_scores[task].get(state, 0)
        current_risk += weights[task] * task_risk_score
    
    # 将当前时间点的风险分数加入结果列表
    combined_risk_scores.append((current_time, current_risk, total_risk_score))

# 将结果转换为DataFrame
combined_risk_df = pd.DataFrame(combined_risk_scores, columns=['Time', 'Current Risk Score', 'Sliding Window Risk Score'])

# 定义颜色映射函数
def risk_to_color(risk):
    cmap = mcolors.LinearSegmentedColormap.from_list('risk_colors', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
    return cmap(risk)

# 添加颜色列
combined_risk_df['Current Risk Color'] = combined_risk_df['Current Risk Score'].apply(risk_to_color)
combined_risk_df['Sliding Window Risk Color'] = combined_risk_df['Sliding Window Risk Score'].apply(risk_to_color)

# 可视化
plt.figure(figsize=(10, 6))

# 当前风险分数
plt.subplot(2, 1, 1)
plt.bar(combined_risk_df['Time'], combined_risk_df['Current Risk Score'], color=combined_risk_df['Current Risk Color'], edgecolor='black')
plt.title('Current Risk Score Per Second')
plt.xlabel('Time (seconds)')
plt.ylabel('Risk Score')

# 滑动窗口风险分数
plt.subplot(2, 1, 2)
plt.bar(combined_risk_df['Time'], combined_risk_df['Sliding Window Risk Score'], color=combined_risk_df['Sliding Window Risk Color'], edgecolor='black')
plt.title('Sliding Window Risk Score (Past 30 Seconds)')
plt.xlabel('Time (seconds)')
plt.ylabel('Risk Score')

plt.tight_layout()
plt.show()

# 导出到单个CSV文件
combined_risk_df.to_csv('/home/users/ntu/chih0001/scratch/VLM/LLaVA/risk_assessment/combined_risk_scores.csv', index=False)
