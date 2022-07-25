# unity-ml-agents-DQN-based
졸업 프로젝트 이후 이것 저것 손보는 중

### (2022년 7월 4일) 거리, 속도 기준 정하기
보통 유니티에서 환경을 구성할 때, 기본 3D object에서 Cube를 기준으로(1m x 1m x 1m) 크기를 결정하게 된다.
내가 구성한 자율주행 프로젝트에서는 각 2배씩 하여 (2m x 2m x 2m)로 기존의 환경보다 8배 크게 구성한다.

* 3D object Cube: 2m x 2m x 2m

* 차량 에이전트 길이: 4.5m

* 차량 에이전트 폭: 2.5m

* 1초에 50step(delta time) 진행

* 차량 속도가 5.0f: step 당 0.2m -> 1초에 10m 이동: 36km/h

* 차량 속도가 7.5f: step 당 0.3m -> 1초에 15m 이동: 54km/h

* 차량 속도가 10.0f: step 당 0.4m -> 1초에 20m 이동: 72km/h

### (2022년 7월 6일) 실제 외곽 순환도로 환경과 유사하게 도로 재구성
실제 외곽 순환 도로와 비슷하게 구성하였으며, 왕복 4차선의 도로이다. 보통 차량이 차선을 달릴 경우 추월할 경우를 제외하고 2차선을 달리도록 되어있는데, 자율 주행의 경우 안전이 최우선이므로 2차선 고정 주행을 할 수 있도록 설정했다.
<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/20fc3687769f0d2dacf8c5f7e0fe4b5801b27239/Image/new%20road%20environment.png"></img>

### (2022년 7월 9일) Bullet through paper 문제 발생
게임에 사용되는 물리학은 연속적이지 않고, 게임 화면을 이용해 충돌을 판단한다. 하지만 object의 이동 속도가 빠르다면 collider가 있음에도 충돌을 인식하지 못하여 통과하게 된다.


1. 도로를 감싸고 있는 주변 연석에 차량이 최고 속력으로 움직여도 벽의 두께를 통과하지 못할 정도로 두께를 두껍게 한다
    

2. 차량 에이전트의 Rigidbody component에서 collision detection을 연속적으로 변경하여 연속적인 충돌 감지를 할 수 있게 한다.

3. 차량 에이전트의 Rigidbody component에서 Interpolate 옵션을 Extrapolate으로 하면, 충돌을 1frame 앞서 예측하여 더 나은 충돌 감지를 할 수 있게 한다.

> 효과를 볼 수 없었음

### (2022년 7월 10일) 발생되는 문제에 대한 내 생각
Bullet through paper 문제를 연속적인 충돌 감지로 해결했다고 생각했지만 해결이 되지 않았다. 어떤 문제인지 알아보기 위해 Gizmos.DrawWireCube 를 이용해 차량 에이전트를 둘러싸고 있는 cube를 그려봤다. 

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/20fc3687769f0d2dacf8c5f7e0fe4b5801b27239/Image/around%20car%20agent.png" width="40%" height="30%"></img>

차량에 설정한 Box Collider Component의 Edit Collider와 달라 이 부분에 의한 문제로 충돌 감지가 제대로 이루어지지 않는 것으로 추측된다. 아무래도 유니티에 대한 기초가 부족하여 구글링하여 나온 자료를 그대로 사용하다 보니 생긴 문제라고 생각된다.

유니티 자체를 내가 잘 이해하지 못하고 프로젝트를 진행하고 있다는 생각이 드는게.. position, localScale 을 box collider의 Center, Size로  변경하면 해결될까 싶어 진행해봤지만, 너무 다른 결과가 나와 혼란스럽다.

유니티 기본 component, 기능들에 대해 제대로 기초부터 공부해 봐야 할 것 같다.

### (2022년 7월 11일) 해결 방법은 찾았는데...
프로젝트에 ‘wall’ tag를 걸어준 연석들 box collider component의 is trigger를 체크하면, 충돌은 감지하지만, 막히지 않고 그대로 통과해버린다. 

기존의 Physics.OverlapBox(에이전트 중심, 에이전트 크기 절반)을 사용했을 경우 충돌 감지만으로 체크를 하여 EndEpisode() 를 할 수 있었지만, 위의 아직 해결하지 못한 문제로 인해 차량의 localScale과 보이는 차량의 크기가 달라 연석에 닿았을 경우 reset 되어야 하지만 그렇지 못하고 그대로 통과해버리는 것을 찾았다. 

OnCollisionEnter() 함수를 사용하여 연석의 is trigger 항목을 체크 해제 하면, collider끼리의 충돌이 발생할 경우 tag를 걸어주어 바로 EndEpisode()를 할 수 있도록 재설계했다.

### (2022년 7월 15일) 도로 환경을 바꾸고 학습이 이루어지지 않는다..
도로 환경이 간단해지면 간단해졌지 좁아지거나 어려운 커브길이 생긴것이 아니다.
하이퍼 파라미터 설계, 에이전트에 주어지는 보상 설계를 다시 해봐야겠다.

1. 이동한 거리에 비례하여 보상 부여: 단위 step 당 속도에 따라 이동한 거리에 보상을 부여하여 결과적으로 빠른 속도로 주행할 수 있도록 한다.

2. 벽(연석)에 부딪힐 경우 큰 (-) 보상 부여: 기존에 설계한 보상에서 벽에 부딪힐 경우 훨씬 큰 (-) 보상을 부여하여 벽에 박으면 큰일이 난다는 것을 인식할 수 있도록 한다.

3. 목표지점에 도착할 경우 큰 (+) 보상 부여: 벽에 부딪혔을 때와 비슷하게 큰 보상을 줌으로서 다시 목표지점에 도달할 수 있도록 한다.

4. 중앙선 넘어서 주행할 경우 복구 못할 정도의 (-) 보상 부여: 벽에 부딪혔을 때 만큼은 아니지만 큰 (-) 보상을 부여하여 다시 정상 주행을 해도 복구 못하도록 한다.

5. Epsilon greedy의 Epsilon-decay 키우기: 기존의 Epsilon greedy 방식에서는 Epsilon이 1에서 0.00001씩 선형적으로 감소하게 설계되어 있었다. 이에 평균적으로 4000 에피소드가 지남에도 랜덤한 행동 선택을 많이 할 것 같아 epsilon decay 크기를 키워 어느정도 에피소드가 지나면 maxQ 값에 따른 행동 선택을 하게끔 한다.

6. Target network update 작게: Target network update 횟수가 기존에는 너무 적어 학습이 제대로 이루어지지 않은 것이라 판단하여 작게한다.

### (2022년 7월 22일) 하이퍼 파라미터 튜닝은 어려워...
많은 경우의 수를 생각하고 변경해가며 학습을 돌리고 있는데, 한번 학습 환경을 돌리는데도 오랜 시간이 걸리며, 제대로 학습이 되는지 확인하려면 마지막 에피소드가 끝날 때까지 기다리고, 그래프로 봐야 하기 때문에 정말 힘든 시간의 연속이다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/41476d889d24c7594dd1bbbcb450f8337b0a7fef/Image/220722-DRL.gif"/></img>

### (2022년 7월25일) 차량 위치 변경
7월 22일 어느정도 학습된 차량의 영상에서 보다시피 어느정도 학습이 된 것을 확인했다.
하지만 우측으로 방향을 틀긴 하지만 좌측으로 방향을 틀지 못하는 것을 확인할 수 있는데, 이는 좌측으로 방향을 틀기까지 긴 시간 주행을 해야하기 때문인 것으로 생각된다.
차량의 도로상 위치를 직선, 우측, 좌측 모두 수행할 수 있게끔 수정하여 학습을 다시 시도해야겠다.

* 차량의 도로상 위치 변경 전
<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/7d094d59744b0775107dca6228e241fad850ffef/Image/before%20car%20agent%20location.PNG" width="50%" height="50%" >

* 차량의 도로상 위치 변경 후
<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/7d094d59744b0775107dca6228e241fad850ffef/Image/move%20car%20agent%20location.PNG" width="50%" height="50%">