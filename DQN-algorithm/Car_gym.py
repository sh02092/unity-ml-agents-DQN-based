from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.environment import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import numpy as np
import mlagents.trainers
import copy


from collections import namedtuple

class Car_gym(object):
    def __init__(self, time_scale=1.0, filename='default', port=11300, width=800, height=600):

        self.engine_configuration_channel = EngineConfigurationChannel()
        print(f"VERSION : {mlagents.trainers.__version__}")
        self.env = UnityEnvironment(
            file_name=filename,
            worker_id=port,
            side_channels=[self.engine_configuration_channel])
        self.env.reset()

        self.behavior_name = list(self.env.behavior_specs)[0]
        self.engine_configuration_channel.set_configuration_parameters(time_scale=time_scale, width=width, height=height, capture_frame_rate=0)

    def reset(self):
        self.env.reset()
        dec, _ = self.env.get_steps(self.behavior_name)

        # print("-----------------------------------------------")
        # print(len(dec.obs))
        # print("--------------------car_gym---------------------------")

        state = [dec.obs[i][0] for i in range(len(dec.obs))]

        # print("-----------------------------------------------")

        # mutable한 list의 특성 때문에 객체 자체를 복사해 원본 배열 보존
        return copy.deepcopy(state)

    def step(self, action):
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.array([[action]]))

        # dimension
        #print("action_tuple dimension : ", np.array([[action]]).shape)

        self.env.set_actions(self.behavior_name, action_tuple)
        
        # # 한 step당 진행하는 deltatime
        # for _ in range(step_deltatime):
        #     self.env.step()

        #     # decision, terminal 
        #     dec, term = self.env.get_steps(self.behavior_name)

        #     done = len(term.agent_id) > 0
        #     reward = term.reward[0] if done else dec.reward[0]

        #     if done:
        #         # episode 종료시 next_state
        #         next_state = [term.obs[i][0] for i in range(len(dec.obs))]
        #         # step_deltatime동안 진행하는 도중 done 되면 break
        #         break
        #     else:
        #         # episode 진행 중 next_state
        #         next_state = [dec.obs[i][0] for i in range(len(dec.obs))]
        
        self.env.step()

        # decision, terminal 
        dec, term = self.env.get_steps(self.behavior_name)

        done = len(term.agent_id) > 0
        reward = term.reward[0] if done else dec.reward[0]

        if done:
            # episode 종료시 next_state
            next_state = [term.obs[i][0] for i in range(len(dec.obs))]
            # step_deltatime동안 진행하는 도중 done 되면 break
        else:
            # episode 진행 중 next_state
            next_state = [dec.obs[i][0] for i in range(len(dec.obs))]


        return copy.deepcopy(next_state), reward, done
    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    env = Car_gym(
        time_scale=1.0,
        filename='Advanced_DQN_220801.exe')
    state = env.reset()
    print(state[1])
    print("-----------------------------------------------")

    next_obs, reward, done = env.step(2, 5)
