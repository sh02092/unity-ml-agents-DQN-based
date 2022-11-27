from asyncore import dispatcher_with_send
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import datetime
from collections import deque
from Car_gym import Car_gym

import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()

        # Ray 추가할때 사용
        # self.vector_fc = nn.Linear(95, 512)

        self.image_cnn = nn.Sequential(
            # 64 x 64 x 4 -> 15 x 15 x 32
            nn.Conv2d(4,32,kernel_size=8,stride=4),
            nn.ReLU(),
            # 15 x 15 x 32 -> 6 x 6 x 64
            nn.Conv2d(32,64,kernel_size=5,stride=2),
            nn.ReLU(),
            # 6 x 6 x 64 -> 4 x 4 x 64
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.ReLU()
        )
        # 4 x 4 x 64 -> 1 x 1024
        # 1 x 1024 + 1 x 5 -> 1 x 1029
        # 1 x 1029 -> 1 x 512 
        self.image_fc = nn.Linear(1029, 512)

        self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, 512)

        self.fc_action1 = nn.Linear(512, 256)
        self.fc_action2 = nn.Linear(256, 15)
    
    def forward(self, camera, ray):
        camera_obs = camera
        ray_obs = ray

        # batch : 1
        batch = camera_obs.size(0)

        # self.image_cnn(camera_obs) : 
        # torch.view(batch, -1) : self.image_cnn(camera_obs) tensor를 (1, ?)의 2차원 tensor로 변경
        # camera_obs 사이즈(image_cnn 이후): torch.Size([1, 64, 4, 4])
        # camera_obs 사이즈(view(batch, -1) 이후) : torch.Size([1, 1024])
        camera_obs = self.image_cnn(camera_obs).view(batch, -1)

        # camera, ray obs를 합치기 위해 torch shape을 (1, ?) -> (?, 1) 로 변경
        # col vector -> row vector화 : torch.cat 사용하기 위해
        camera_obs = camera_obs.view(-1, batch)
        ray_obs = ray_obs.view(-1, batch)

        # x에 ray 성분 추가
        # torch.Size([1, 1029])
        x = torch.cat([camera_obs, ray_obs], dim=0)   
        # row vector -> col vector화 : fully connected layer 적용하기 위해
        # torch.Size([1029, 1])
        x = x.view(batch, -1)

        # print("현재 x tensor의 shape : ", x.shape)
        # print("-------------------------------------")

        x = self.image_fc(x)
        # print("현재 camera sensor : ", x.shape)
        # print("----------------------------------")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logic = F.relu(self.fc_action1(x))
        Q_values = self.fc_action2(action_logic)

        # 15개 action에 대한 Q values
        # print("Q_values : ", Q_values)

        return Q_values

class Agent:
    def __init__(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy_net = Q_network().to(device)
        self.Q_target_net = Q_network().to(device)

        self.learning_rate = 0.0002

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Q의 nn parameter를 target network에 불러오기
        self.Q_target_net.load_state_dict(self.policy_net.state_dict())

        # epsilon
        self.epsilon = 1
        # epsilon decay 값
        self.epsilon_decay = 0.00001

        self.device = device
        
        self.data_buffer = deque(maxlen=100000)
        #self.data_buffer = deque(maxlen=30000)

        self.init_check = 0
        self.cur_state = None

        self.x_epoch = list()
        # max Q value 32개(batch size) 평균
        self.y_max_Q_avg = list()

        self.y_loss = list()

    def epsilon_greedy(self, Q_values):
        # 난수 생성, 
        # epsilon보다 작은 값일 경우
        if np.random.rand() <= self.epsilon:
            # action을 random하게 선택
            # action = random.randrange(15)
            action = random.randrange(15)

            return action

        # epsilon보다 큰 값일 경우
        else:
            # 학습된 Q value 값중 가장 큰 action 선택
            return Q_values.argmax().item()
    
    # def policy(self, obs):
    # def train_policcy(self, obs):

    # model 저장
    def save_model(self):
        torch.save(self.policy_net.state_dict(), 'best_model/best_model.pkl')
        return None

    # model 불러오기
    def load_model(self):
        self.policy_net.load_state_dict(torch.load('best_model/best_model.pkl', map_location=self.device))
        return None

    def store_trajectory(self, traj):
        self.data_buffer.append(traj)

    # 1. resizing : 64 * 64, gray scale로
    def re_scale_frame(self, obs):
        return resize(rgb2gray(obs), (64,64))

    # 2. image 4개씩 쌓기
    def init_image_obs(self, obs):
        # image 4개씩
        obs = self.re_scale_frame(obs)
        frame_obs = [obs for _ in range(4)]
        frame_obs = np.stack(frame_obs, axis = 0)

        return frame_obs

    # 3. 4장 쌓인 Image return
    def init_obs(self, obs):
        return self.init_image_obs(obs)
    
    def camera_obs(self, obs):
        # 4차원의 땡 * 땡 * 땡 * 땡 인지...
        camera_obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        # print("camera_obs 함수 안에서 camera shape : ", camera_obs.shape)
        # print("----------------------------------")
        return camera_obs
    
    def ray_obs(self, obs):
        # ray_obs 사이즈 : torch.Size([5]) -> torch.Size([1, 5])
        ray_obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        # print("ray_obs 함수 안에서 ray shape : ", ray_obs.shape)
        # print("----------------------------------")
        return ray_obs

    # numpy 변환은 cpu 연산으로 한 결과에만 적용 가능?
    def ray_obs_cpu(self, obs):
        obs = torch.Tensor(obs).unsqueeze(0)
        return obs.numpy()

    # FIFO, 4개씩 쌓기
    # step 증가함에 따라 맨 앞 frame 교체
    # 1111 -> 1112 -> 1123 -> 1234 -> 2345 -> ...
    def accumulated_image_obs(self, obs, new_frame):
        temp_obs = obs[1:,:,:]
        new_frame = np.expand_dims(self.re_scale_frame(new_frame), axis=0)
        frame_obs = np.concatenate((temp_obs, new_frame), axis=0)
        # print(temp_obs)
        # print(np.shape(frame_obs))
        # print('-------------------------------')
        return frame_obs

    def accumlated_all_obs(self, obs, next_obs):
        return self.accumulated_image_obs(obs, next_obs)

    def convert_action(self, action):
        # if action == 0: return 0
        # if action == 1: return 1
        # if action == 2: return 2
        # if action == 3: return 3
        # if action == 4: return 4
        # if action == 5: return 5
        # if action == 6: return 6
        # if action == 7: return 7
        # if action == 8: return 8
        # if action == 9: return 9
        # if action == 10: return 10
        # if action == 11: return 11
        # if action == 12: return 12
        # if action == 13: return 13
        # if action == 14: return 14

        return action

    # action 선택, discrete action 15개 존재
    # obs shape : torch.Size([1, 4, 64, 64])
    def train_policy(self, obs_camera, obs_ray):
        Q_values = self.policy_net(obs_camera, obs_ray)
        action = self.epsilon_greedy(Q_values)
        return self.convert_action(action), action
        
    def batch_torch_obs(self, obs):
        # 3차원 (1, 84, 84, 3) -> 2차원 (84, 84, 3)
        return torch.Tensor(np.stack([obs], axis=0)).squeeze(dim=0).to(self.device)

    def batch_ray_obs(self, obs):
        # 1차원 -> 2차원(1 , 5)
        return torch.Tensor(np.stack([obs], axis=0)).unsqueeze(0).to(self.device)

    # update target network 
    # 저장되어있던 를 target network에 복사
    def update_target(self):
        self.Q_target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, step, update_target):

        # loss_write = SummaryWriter(logdir='loss_value_1')
        # if step % update_target == 0:
        #     loss_write.add_scalar('step', step, step)
        #     loss_write.add_scalar('loss', )


        # discount factor
        gamma = 0.9

        # epsilon decaying
        self.epsilon -= self.epsilon_decay
        # min of epsilon : 0.05
        self.epsilon = max(self.epsilon, 0.05)
        # batch_size 만큼 random sampling
        # data_buffer(경험)에서 32개 data random하게 select
        
        # random_mini_batch = random.sample(self.data_buffer, 32)
        random_mini_batch = random.sample(self.data_buffer, 64)

        # data 분배
        # 현재 state, action, reward, next state, 게임종료
        obs_camera_list, obs_ray_list, action_list, reward_list, next_obs_camera_list, next_obs_ray_list, mask_list = [], [], [], [], [], [], []

        for all_obs in random_mini_batch:
            s_c, s_r, a, r, next_s_c, next_s_r, mask = all_obs
            obs_camera_list.append(s_c)
            obs_ray_list.append(s_r)
            action_list.append(a)
            reward_list.append(r)
            next_obs_camera_list.append(next_s_c)
            next_obs_ray_list.append(next_s_r)
            mask_list.append(mask)

        # tensor
        obses_camera = self.batch_torch_obs(obs_camera_list)
        obses_ray = self.batch_ray_obs(obs_ray_list)

        actions = torch.LongTensor(action_list).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(reward_list).to(self.device)
        
        next_obses_camera = self.batch_torch_obs(next_obs_camera_list)
        next_obses_ray = self.batch_ray_obs(next_obs_ray_list)

        masks = torch.Tensor(mask_list).to(self.device)


        # get Q-value
        Q_values = self.policy_net(obses_camera, obses_ray)
        # print(Q_values)
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++')

        # 추정값
        q_value = Q_values.gather(1, actions).view(-1)
        # print(q_value)
        # print('--------------------------------------')
        
        # get target, y(타겟값) 구하기 위한 다음 state에서의 max Q value
        # target network에서 next state에서의 max Q value -> 상수값
        target_q_value = self.Q_target_net(next_obses_camera, next_obses_ray).max(1)[0]
        

        # 타겟값(y)
        Y = rewards + masks * gamma * target_q_value

        # loss 정의
        MSE = nn.MSELoss()
        #           input,  target
        loss = MSE(q_value, Y.detach())

        # 10,000번의 episode동안 몇 번의 target network update가 있는지
        # target network update 마다 max Q-value / loss function 분포
        if step % update_target == 0:
            self.x_epoch.append(step//update_target)

            # tensor -> list
            # max Q-value 분포
            tensor_to_list_q_value = target_q_value.tolist()
            # max_Q 값들(batch size : 32개)의 평균 값 
            list_q_value_avg = sum(tensor_to_list_q_value)/len(tensor_to_list_q_value)
            self.y_max_Q_avg.append(list_q_value_avg)

            # loss 평균 분포(reduction = mean)
            loss_in_list = loss.tolist()
            self.y_loss.append(loss_in_list)

        # backward 시작
        # Adam optimizer를 통해 Q-network의 model.parameters()의 loss function 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    engine_configuration_channel = EngineConfigurationChannel()
    env = Car_gym(
        time_scale=1.0, port=11300, filename='Advanced_DQN_220801.exe')
    score = 0
    reward_score = 0
    # episode당 step
    episode_step = 0
    # 전체 누적 step
    step = 0

    action_speed = 0

    # 한 step당 진행될 deltatime
    step_deltatime = 5

    initial_exploration = 200
    
    update_target = 2000

    # episode당 이동한 거리
    episode_distance = 0
    total_epi_dis = 0
    y_epi_dis = list()

    x_episode = list()
    total_step = 0
    y_epi_avg =list()



    # x축 : 학습 과정에서 epoch 수, y축 : maxQ 값의 변화
    # x축 : step 수                y축 : loss 값
    # x축 : episode 수,            y축 : episode당 step수(주행시간)

    # Agent() class 초기화: __init__(self) 자동으로 호출
    agent = Agent()

    for epi in range(7001):

        speed_count_1 = 0
        speed_count_2 = 0   
        speed_count_3 = 0

        obs = env.reset()
        # 3차원 (1, 84, 84, 3) : camera sensor
        obs_camera = obs[0]


        obs_camera = torch.Tensor(obs_camera)
        # print("main 함수에서 camera tensor의 shape : ", obs_camera.shape)
        # print("-------------------------------------")        
        
        # 3차원 (1, 84, 84, 3) -> 2차원 (84, 84, 3)
        obs_camera = torch.Tensor(obs_camera).squeeze(dim=0)
        # (84, 84, 3) -> (64, 64, 1) -> 4장씩 쌓아 (64, 64, 4)
        # 같은 Image 4장 쌓기 -> 이후 action에 따라 환경이 바뀌고, 다른 Image data 쌓임
        obs_camera = agent.init_obs(obs_camera)


        # 1차원                : ray sensor
        obs_ray = obs[1]
        obs_ray_tensor = [obs_ray[1], obs_ray[7], obs_ray[9], obs_ray[35], obs_ray[37]]
        obs_ray_tensor = torch.Tensor(obs_ray_tensor)
        # 0 3 4 17 18
        # obs_ray[1] : 정면
        # obs_ray[7] : 우측 대각
        # obs_ray[9] : 좌측 대각
        # obs_ray[35] : 우측
        # obs_ray[37] : 좌측
        
        # print("main 함수에서 ray tensor의 shape : ", obs_ray.shape)
        # print("-------------------------------------")
        
        # obs_ray = agent.ray_obs(obs_ray)

        while True:
            # print('누적 step: ', step)

            # action 선택
            action, dis_action = agent.train_policy(agent.camera_obs(obs_camera), agent.ray_obs(obs_ray_tensor))
            # action에 따른 step()
            # next step, reward, done 여부
            next_obs, reward, done = env.step(action)

            # print('reward : %f'% reward)

            # action당 이동 거리 측정 -> 한 Episode당 이동한 거리 측정
            if not done:
                if action == 0 or action == 3 or action == 6 or action == 9 or action ==  12:    
                    episode_distance += 1.0
                    speed_count_1 += 1
                elif action == 1 or action == 4 or action == 7 or action == 10 or action ==  13: 
                    episode_distance += 1.5
                    speed_count_2 += 1
                elif action == 2 or action == 5 or action == 8 or action == 11 or action ==  14: 
                    episode_distance += 2.0
                    speed_count_3 += 1
            else:
                if epi % 100 == 0:
                    y_epi_dis.append(total_epi_dis // 100)
                    total_epi_dis = 0
                else:
                    total_epi_dis += episode_distance

            # state는 camera sensor로 얻은 Image만
            next_obs_camera = next_obs[0]
            next_obs_ray = next_obs[1]

            #print(next_obs_ray.shape)
            next_obs_ray_tensor = [next_obs_ray[1], next_obs_ray[7], next_obs_ray[9], next_obs_ray[35], next_obs_ray[37]]
            next_obs_ray_tensor = torch.Tensor(next_obs_ray_tensor)

            next_obs_camera = torch.Tensor(next_obs_camera).squeeze(dim=0)
            # step이 증가함에 따라 4장 중 1장씩 밀기(FIFO)
            next_obs_camera = agent.accumlated_all_obs(obs_camera, next_obs_camera)


            mask = 0 if done else 1
            # print("%d번째 step에서의 reward : %f, action speed : %f"%(step, reward, action_speed))
            # if reward == 0:
            #     print("엥...")
            score += reward
            reward_score += reward

            # maxlen = 100,000인 data buffer(경험 데이터)에 저장
            # 현재 상태(camera, ray), 현재 행동, 보상, 다음 상태(camera, ray), 종료 유/무
            agent.store_trajectory((obs_camera, agent.ray_obs_cpu(obs_ray_tensor), dis_action, reward, next_obs_camera, agent.ray_obs_cpu(next_obs_ray_tensor), mask))

            obs_camera = next_obs_camera
            obs_ray_tensor = next_obs_ray_tensor

            # step > 2000
            if step > initial_exploration:
                agent.train(step, update_target)
                # 1,000 step마다 모델 저장
                if step % 1000 == 0:
                    agent.save_model()
                # update_target 마다 저장된 모델을 타겟 네트워크로 불러오기
                if step % update_target == 0:
                    agent.update_target()

            episode_step += 1
            step += 1

            if done:
                break

        if (epi + 1) % 1 == 0:
            print('%d 번째 episode의 총 step: %d'% (epi + 1, episode_step))
            print('Speed  5.0: %d 번'% speed_count_1)
            print('Speed  7.5: %d 번'% speed_count_2)
            print('Speed 10.0: %d 번'% speed_count_3)
            print('True_score: %f\n'% score)

        # 100 episode까지의 step 전체
        total_step = total_step + episode_step
        # 100 episode 마다
        if epi % 100 == 0:
            x_episode.append(epi)
            # second의 평균
            y_epi_avg.append(total_step//100)
            total_step = 0

        score = 0
        reward_score = 0
        episode_step = 0
        episode_distance = 0
    # 100 episode 마다 episode 종료시까지의 평균 주행시간(second)
    plt.figure(1)
    plt.plot(x_episode, y_epi_avg)
    plt.xlabel('episode')
    plt.ylabel('driving time(sec)')

    # target update(2000 step) 마다 maxQ 값 변화
    plt.figure(2)
    plt.plot(agent.x_epoch, agent.y_max_Q_avg)
    plt.xlabel('target update')
    plt.ylabel('max Q value')

    # target update(2000 step) 마다 loss 값 변화
    plt.figure(3)
    plt.plot(agent.x_epoch, agent.y_loss)
    plt.xlabel('target update')
    plt.ylabel('loss')

    # 100 episode마다 평균 주행시간
    plt.figure(4)
    plt.plot(x_episode, y_epi_dis)
    plt.xlabel('episode')
    plt.ylabel('move distance')
    plt.show()

    # target update(2000 step) 마다 loss 값 변화
    


if __name__ == '__main__':
    main()


# Cube 크기 : 2m x 2m

# Plane 가로 : 200m
# Plane 세로 : 200m
# 도로 타일 1칸당 2m

# 차량 길이 : 약 4.5m
# 차량 폭   : 약 2.5m

# 전체 도로 : 약 660m
# 차 대비 약 147배

# 1초 : 50 step
# moveSpeed =  5.0f일 때 한 step당 0.2m 이동 : 1초에 10m 이동 -> 36km/h
# moveSpeed =  7.5f일 때 한 step당 0.3m 이동 : 1초에 15m 이동 -> 54km/h
# moveSpeed = 10.0f일 때 한 step당 0.4m 이동 : 1초에 20m 이동 -> 72km/h

# learning rate -> 0.0002
# mini batch size -> 64

# (22/11/11)
# Unity: 4번의 step마다 action 요청 -> Unity에서의 4step == Python에서의 1step
# Python: 10의 속도로 약 1250번의 action 요청 후 목표지점 도달
# --> 전체 도로 길이 = 1250 x 4 x 0.4 = 약 2.0 km