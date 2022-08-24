### (2022년 8월 24일) Obstacle 차량 추가

도로에 장애물 차량을 추가하여 Agent 차량이 학습을 할 때 장애물에 대한 대처를 확인한다.
장애물 차량의 경우 하드코딩으로 구성되며, 각 차선에 여러대 놓고, 속도를 일정하게 유지한다.
Agent와 부딪힐 경우 Agent는 Reset 되며, 장애물끼리 부딪힐 경우에는 그대로 통과하게끔 설계한다.

    'wall', 'goal', 'obstacle' Tag의 Box Collider Is Trigger: 체크
    Agent의 Box Collider Is Trigger: 체크 안함

따라서 Agent는 'wall', 'goal', 'obstacle' Tag 모두 부딪힐 경우 Reset 되고, 
'obstacle' 끼리 부딪힐 경우 그대로 통과한다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/f53449807343473ac7d9d7d2a1ca9cd13ef041ec/Image/add%20ray%20sensor.jpg" width="50%" height="50%">
