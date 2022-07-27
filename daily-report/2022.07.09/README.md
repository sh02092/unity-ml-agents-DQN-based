### (2022년 7월 9일) Bullet through paper 문제 발생
게임에 사용되는 물리학은 연속적이지 않고, 게임 화면을 이용해 충돌을 판단한다. 하지만 object의 이동 속도가 빠르다면 collider가 있음에도 충돌을 인식하지 못하여 통과하게 된다.


1. 도로를 감싸고 있는 주변 연석에 차량이 최고 속력으로 움직여도 벽의 두께를 통과하지 못할 정도로 두께를 두껍게 한다
    

2. 차량 에이전트의 Rigidbody component에서 collision detection을 연속적으로 변경하여 연속적인 충돌 감지를 할 수 있게 한다.

3. 차량 에이전트의 Rigidbody component에서 Interpolate 옵션을 Extrapolate으로 하면, 충돌을 1frame 앞서 예측하여 더 나은 충돌 감지를 할 수 있게 한다.

> 효과를 볼 수 없었음