### (2022년 7월 25일) 차량 위치 변경
7월 22일 어느정도 학습된 차량의 영상에서 보다시피 어느정도 학습이 된 것을 확인했다.
하지만 우측으로 방향을 틀긴 하지만 좌측으로 방향을 틀지 못하는 것을 확인할 수 있는데, 이는 좌측으로 방향을 틀기까지 긴 시간 주행을 해야하기 때문인 것으로 생각된다.
차량의 도로상 위치를 직선, 우측, 좌측 모두 수행할 수 있게끔 수정하여 학습을 다시 시도해야겠다.

* 차량의 도로상 위치 변경 전
<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/7d094d59744b0775107dca6228e241fad850ffef/Image/before%20car%20agent%20location.PNG" width="80%" height="80%" >

* 차량의 도로상 위치 변경 후
<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/7d094d59744b0775107dca6228e241fad850ffef/Image/move%20car%20agent%20location.PNG" width="80%" height="80%">


### (2022년 7월 26일) 결과 분석
7월 25일 학습 시켰던 차량을 10,000번의 에피소드 거친 이후 결과를 확인했다. 

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/dab711a1f125e8a757988684932ea3831affb4e4/Image/max%20Q-value%20220725.PNG" width="50%" height="50%">

약 80~100 번의 타겟 업데이트 이후 점점 max Q-value의 값이 작아지는 것을 볼 수 있다. 
위 결과에 대한 결과 분석

* Epsilon greedy: 약 90번의 타겟 업데이트를 할 때까지 입실론 그리디의 입실론 값을 최솟값에 수렴하게 하여 더이상 랜덤한 행동을 하지 않도록 바꾸면 어떨까라는 생각이 들어서 계산해본 결과 90 x 1,000(타겟 업데이트 주기) = 900,000 스탭으로 충분히 입실론 값이 최솟값인 0.05에 도달하는데는 문제없는 스텝이다. 

* Overfitting: 너무 모델이 반복적인 초반 도로에만 최적화되게끔 학습이 된 것인지 확인하기 위해 약 90번의 타겟 업데이트 이후 타겟 네트워크를 고정시켜 다시 학습해봐야겠다.

* 차량 에이전트가 한 가지 행동을 5 스텝동안 지속: 2차선에서 달리면서 차량이 학습을 하는 과정에서 핸들을 오른쪽으로 꺾고 5스텝을 진행하면 5스텝이 다하기 전에 리셋이 되지는 않지만 다시 복구할 수 없을 정도로 벽과 가까워지게 되어 곧바로 리셋된다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/dab711a1f125e8a757988684932ea3831affb4e4/Image/driving%20time%20220725.PNG" width="50%" height="50%">

100 에피소드마다 평균치를 내어 차량 에이전트가 주행한 시간을 측정했는데, 꾸준히 증가하며 학습이 완료된 시점에 수렴하는 것이 아니라 가늠하기 힘들 정도로 심하게 편차가 큰 것을 볼 수 있다.

loss 값을 따로 뽑아 max Q-value 값과 비교, 차량 에이전트가 한 가지 행동을 3 스텝동안 지속하도록 변경하여 기존의 10,000번의 에피소드가 아닌 7,000번의 에피소드로 줄여 학습을 진행한다.


### (2022년 7월 27일) 불규칙한 loss
7월 26일 학습 시켰던 차량의 7,000번 에피소드 거친 이후 결과를 확인했다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/bc67e6ab40bbd878f0826ef00bfc4951b6b02921/Image/loss%20220727.PNG" width="50%" height="50%">

loss 값들의 평균치를 전체 에피소드에 대해 뽑아봤는데, 결과를 분석할 수 없을 정도로 너무 혼란스러운 결과가 출력되었다.
다만 target network가 업데이트할 때마다 loss 값이 증가한 후 target network 업데이트하기 직전까지 감소하는 것으로 보아 코드상의 문제점으로 보이진 않았다.

target network의 업데이트 과정에서 이전보다 개선된 일반 네트워크에 대한 복제가 이루어져야 하는데 학습자체가 잘 이루어지지 않아 일어난 문제일 것 같았다.
목표지점인 goal까지 가려면 한 바퀴를 모두 돌아야 하는데.. 차량이 학습을 하기에 도로 환경이 너무 넓다는 생각이 들었다.
차량이 한 바퀴 전체를 돌게끔 학습을 하려면 중간중간 체크포인트를 두어 잘 주행하고 있다는 것을 알려줄 수 있는 포인트들을 추가하여 다시 학습을 진행한다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/bc67e6ab40bbd878f0826ef00bfc4951b6b02921/Image/road%20with%20check%20point.PNG">