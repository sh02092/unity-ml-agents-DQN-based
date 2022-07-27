### (2022년 7월 27일) 불규칙한 loss
7월 26일 학습 시켰던 차량의 7,000번 에피소드 거친 이후 결과를 확인했다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/bc67e6ab40bbd878f0826ef00bfc4951b6b02921/Image/loss%20220727.PNG" width="50%" height="50%">

loss 값들의 평균치를 전체 에피소드에 대해 뽑아봤는데, 결과를 분석할 수 없을 정도로 너무 혼란스러운 결과가 출력되었다.
다만 target network가 업데이트할 때마다 loss 값이 증가한 후 target network 업데이트하기 직전까지 감소하는 것으로 보아 코드상의 문제점으로 보이진 않았다.

target network의 업데이트 과정에서 이전보다 개선된 일반 네트워크에 대한 복제가 이루어져야 하는데 학습자체가 잘 이루어지지 않아 일어난 문제일 것 같았다.
목표지점인 goal까지 가려면 한 바퀴를 모두 돌아야 하는데.. 차량이 학습을 하기에 도로 환경이 너무 넓다는 생각이 들었다.
차량이 한 바퀴 전체를 돌게끔 학습을 하려면 중간중간 체크포인트를 두어 잘 주행하고 있다는 것을 알려줄 수 있는 포인트들을 추가하여 다시 학습을 진행한다.

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/bc67e6ab40bbd878f0826ef00bfc4951b6b02921/Image/road%20with%20check%20point.PNG">