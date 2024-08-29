import pandas as pd
import random

# 定义任务和状态的可能性
tasks = {
    'Distraction Detection': ['Normal Driving', 'Drinking', 'Phoning Left', 'Phoning Right', 
                              'Texting Left', 'Texting Right', 'Touching Hairs & Makeup', 
                              'Adjusting Glasses', 'Reaching Behind', 'Dropping (Fatigue)'],
    'Emotion Recognition': ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise'],
    'Fatigue Detection': ['Non-Drowsy', 'Drowsy']
}

# 设置生成数据的总时长（秒）
total_duration = 1800  # 例如：3600=1小时，可以根据需要调整

# 生成随机驾驶状态数据
def generate_random_driver_states(total_duration):
    data = []
    current_time = 0

    while current_time < total_duration:
        # 随机选择任务和对应的状态
        task = random.choice(list(tasks.keys()))
        state = random.choice(tasks[task])
        
        # 随机生成该状态的持续时间（1到10秒之间）
        duration = random.randint(1, 10)
        
        # 确保不超过总时长
        if current_time + duration > total_duration:
            duration = total_duration - current_time
        
        # 将数据添加到列表
        data.append([current_time, task, state, duration])
        
        # 更新当前时间
        current_time += duration

    return data

# 生成数据
driver_states = generate_random_driver_states(total_duration)

# 创建DataFrame并保存为CSV文件
df = pd.DataFrame(driver_states, columns=['timestamp', 'task', 'state', 'duration'])
df.to_csv('/home/users/ntu/chih0001/scratch/VLM/LLaVA/risk_assessment/driver_states.csv', index=False)

print(f"Random driver states data generated for {total_duration} seconds and saved to 'driver_states.csv'.")
