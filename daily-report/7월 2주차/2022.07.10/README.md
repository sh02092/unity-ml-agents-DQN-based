### (2022년 7월 10일) 발생되는 문제에 대한 내 생각
Bullet through paper 문제를 연속적인 충돌 감지로 해결했다고 생각했지만 해결이 되지 않았다. 어떤 문제인지 알아보기 위해 Gizmos.DrawWireCube 를 이용해 차량 에이전트를 둘러싸고 있는 cube를 그려봤다. 

<img src="https://github.com/sh02092/unity-ml-agents-DQN-based/blob/20fc3687769f0d2dacf8c5f7e0fe4b5801b27239/Image/around%20car%20agent.png" width="40%" height="30%">

차량에 설정한 Box Collider Component의 Edit Collider와 달라 이 부분에 의한 문제로 충돌 감지가 제대로 이루어지지 않는 것으로 추측된다. 아무래도 유니티에 대한 기초가 부족하여 구글링하여 나온 자료를 그대로 사용하다 보니 생긴 문제라고 생각된다.

유니티 자체를 내가 잘 이해하지 못하고 프로젝트를 진행하고 있다는 생각이 드는게.. position, localScale 을 box collider의 Center, Size로  변경하면 해결될까 싶어 진행해봤지만, 너무 다른 결과가 나와 혼란스럽다.

유니티 기본 component, 기능들에 대해 제대로 기초부터 공부해 봐야 할 것 같다.