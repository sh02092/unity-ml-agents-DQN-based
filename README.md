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