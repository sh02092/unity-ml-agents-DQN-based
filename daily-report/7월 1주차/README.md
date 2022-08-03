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

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/20fc3687769f0d2dacf8c5f7e0fe4b5801b27239/Image/new%20road%20environment.png" width="50%" height="50%">


### (2022년 7월 9일) Bullet through paper 문제 발생
게임에 사용되는 물리학은 연속적이지 않고, 게임 화면을 이용해 충돌을 판단한다. 하지만 object의 이동 속도가 빠르다면 collider가 있음에도 충돌을 인식하지 못하여 통과하게 된다.

1. 도로를 감싸고 있는 주변 연석에 차량이 최고 속력으로 움직여도 벽의 두께를 통과하지 못할 정도로 두께를 두껍게 한다    

2. 차량 에이전트의 Rigidbody component에서 collision detection을 연속적으로 변경하여 연속적인 충돌 감지를 할 수 있게 한다.

3. 차량 에이전트의 Rigidbody component에서 Interpolate 옵션을 Extrapolate으로 하면, 충돌을 1frame 앞서 예측하여 더 나은 충돌 감지를 할 수 있게 한다.

> 효과를 볼 수 없었음