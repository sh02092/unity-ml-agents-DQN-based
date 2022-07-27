### (2022년 7월 11일) 해결 방법은 찾았는데...
프로젝트에 ‘wall’ tag를 걸어준 연석들 box collider component의 is trigger를 체크하면, 충돌은 감지하지만, 막히지 않고 그대로 통과해버린다. 

기존의 Physics.OverlapBox(에이전트 중심, 에이전트 크기 절반)을 사용했을 경우 충돌 감지만으로 체크를 하여 EndEpisode() 를 할 수 있었지만, 위의 아직 해결하지 못한 문제로 인해 차량의 localScale과 보이는 차량의 크기가 달라 연석에 닿았을 경우 reset 되어야 하지만 그렇지 못하고 그대로 통과해버리는 것을 찾았다. 

OnCollisionEnter() 함수를 사용하여 연석의 is trigger 항목을 체크 해제 하면, collider끼리의 충돌이 발생할 경우 tag를 걸어주어 바로 EndEpisode()를 할 수 있도록 재설계했다.