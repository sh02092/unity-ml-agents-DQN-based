using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Rendering;
using UnityEngine.Serialization;

public class CarAgent : Agent
{
    private Transform tr;
    private Rigidbody rb;

    public float moveSpeed;
    public float Turn;
    Vector3 startPosition;
    Vector3 startRotation;

    private float distanceTraveled = 0;
    Vector3 currentPosition;

    private int numStep = 0;

    public override void Initialize()
    {
        tr = this.GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
    }

    public override void OnEpisodeBegin()
    {
       
        // 물리엔진 초기화
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        //tr.localPosition = startPosition;
        //tr.localEulerAngles=startRotation;
        // 위치 초기화
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        // 여기부터

        //transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

        //Debug.Log("position : " + tr.position);
        //Debug.Log("Rotation : " + tr.eulerAngles);
        //Debug.Log(" position : " + transform.position);
        //Debug.Log("red.. scale : " + transform.localScale);
        //Debug.Log("box collider center : " + GetComponentInChildren<BoxCollider>().center);
        //Debug.Log("box collider size : " + GetComponentInChildren<BoxCollider>().size);
        
        // 여기까지 필요없는 것
    }

    public override void CollectObservations(VectorSensor sensor)
    {

    }

    //private void OnDrawGizmos()
    //{
    //    Gizmos.color = Color.red;
    //    Gizmos.DrawWireCube(this.transform.position, GetComponentInChildren<BoxCollider>().size);
    //    //Gizmos.DrawWireCube(GetComponentInChildren<BoxCollider>().center, GetComponentInChildren<BoxCollider>().size);
    //}

    public override void OnActionReceived(ActionBuffers actions)
    {
        var action = actions.DiscreteActions[0];

        // action 적용 전 agent 위치
        currentPosition = transform.position;

        switch (action)
        {
            case 0: Turn = 0f; moveSpeed = 10.0f; break;
            case 1: Turn = 0f; moveSpeed = 7.5f; break;
            case 2: Turn = 0f; moveSpeed = 10.0f; break;
            case 3: Turn = -1.0f; moveSpeed = 5.0f; break;
            case 4: Turn = -1.0f; moveSpeed = 7.5f; break;
            case 5: Turn = -1.0f; moveSpeed = 10.0f; break;
            case 6: Turn = 1.0f; moveSpeed = 5.0f; break;
            case 7: Turn = 1.0f; moveSpeed = 7.5f; break;
            case 8: Turn = 1.0f; moveSpeed = 10.0f; break;
            case 9: Turn = -3.0f; moveSpeed = 5.0f; break;
            case 10: Turn = -3.0f; moveSpeed = 7.5f; break;
            case 11: Turn = -3.0f; moveSpeed = 10.0f; break;
            case 12: Turn = 3.0f; moveSpeed = 5.0f; break;
            case 13: Turn = 3.0f; moveSpeed = 7.5f; break;
            case 14: Turn = 3.0f; moveSpeed = 10.0f; break;
        }
        // action 적용
        transform.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward);
        transform.Rotate(new Vector3(0f, Turn, 0f));


        // new Vector3(0.8575f/2, 0.5201f/2, 2.2634f/2)

        // 여기서부터

        // Physics.OverlapBox(에이전트 중심, 에이전트 box collider 크기의 절반)
        //Collider[] hitObjects = Physics.OverlapBox(
        //    transform.position, transform.localScale / 2);
        ////Collider[] hitObjects = Physics.OverlapBox(
        ////    this.GetComponent<BoxCollider>().center, this.GetComponent<BoxCollider>().size / 2);
        //if (hitObjects.Where(col => col.gameObject.CompareTag("wall")).ToArray().Length == 1)
        //{
        //    AddReward(-30.0f);
        //    Debug.Log("Collision!!");
        //    EndEpisode();
        //}
        //if (hitObjects.Where(col => col.gameObject.CompareTag("notGoal")).ToArray().Length == 1)
        //{
        //    AddReward(-30.0f);
        //    EndEpisode();
        //}
        //if (hitObjects.Where(col => col.gameObject.CompareTag("goal")).ToArray().Length == 1)
        //{
        //    AddReward(50.0f);
        //    EndEpisode();
        //}

        // 여기까지 필요없는 부분.. Physics.OverlapBox 말고 OnCollisionEnter 사용

        // action 적용 후 agent 이동한 거리
        // 차량 속도 5.0f  -> 약 0.1
        // 차량 속도 7.5f  -> 약 0.15
        // 차량 속도 10.0f -> 약 0.2
        distanceTraveled = Vector3.Distance(transform.position, currentPosition);
        //Debug.Log("distance traveled : " + distanceTraveled);

        // action 적용 후 이동한 거리에 따른 보상 부여
        // 차량이 약 0.2 이동했을 경우(차량 속도 10.0f) 보상 부여
        if (distanceTraveled > 0.17) AddReward(0.5f);


        // 레이 센서
        float frontDist = 0;
        float leftDist = 0;
        float rightDist = 0;
        //float dist = 0;
        // 정면 : rayIndex = 0
        // 오른쪽 : rayIndex = 1
        // 왼쪽 : rayIndex = 2
        // Ray Length = 4
        // 좌우 ray length = 4 * 1.5 = 6
        var raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();
        var input = raySensorComponent.GetRayPerceptionInput();
        var outputs = RayPerceptionSensor.Perceive(input);
        for (var rayIndex = 0; rayIndex < outputs.RayOutputs.Length; rayIndex++)
        {
            var extents = input.RayExtents(rayIndex);
            Vector3 startPositionWorld = extents.StartPositionWorld;
            Vector3 endPositionWorld = extents.EndPositionWorld;



            var rayOutput = outputs.RayOutputs[rayIndex];
            if (rayOutput.HasHit)
            {
                Vector3 hitPosition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);
                //Debug.DrawLine(startPositionWorld, hitPosition, Color.red);
                //Debug.Log(rayIndex + " " + Vector3.Distance(hitPosition, startPositionWorld));

                if (rayIndex == 0) frontDist = Vector3.Distance(hitPosition, startPositionWorld);
                else if (rayIndex == 1) rightDist = Vector3.Distance(hitPosition, startPositionWorld);
                else if (rayIndex == 2) leftDist = Vector3.Distance(hitPosition, startPositionWorld);


            }
            else
            {
                if (rayIndex == 0) frontDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                else if (rayIndex == 1) rightDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                else if (rayIndex == 2) leftDist = Vector3.Distance(endPositionWorld, startPositionWorld);
                //Debug.Log(rayIndex + " " + Vector3.Distance(endPositionWorld, startPositionWorld));
            }
        }

        // action 적용 후 레이 센서에 따른 보상 적용
        // 정상 주행
        if (rightDist > 0.6864f && rightDist <= 1.2864f) AddReward(0.6f);
        // 차선 변경 후 주행
        else if (rightDist > 1.2864f && rightDist <= 2.9264f) AddReward(0.1f);
        // 차선 밟기
        // else if (rightDist > 1.2864f && rightDist <= 3.3764f) AddReward(0.2f);
        // 중앙선 침범
        else if (rightDist > 3.3764f) AddReward(-1f);

        //dist = leftDist - rightDist;
        ////Debug.Log("Ray distance : " + dist);
        //// 흰선 밟았을 때
        //if (dist > 4.8084f)                         AddReward(-0.04f);
        //else if (dist > 3.2392f && dist <= 4.8084f) AddReward(-0.02f);
        //else if (dist > 2.7203f && dist <= 3.2392f) AddReward(-0.01f);
        //// 정상 주행
        //else if (dist > 2.2075f && dist <= 2.7203f) AddReward(0.005f);
        //// 중앙선 밟았을 때
        //else if (dist > 2.0387f && dist <= 2.2075f) AddReward(-0.03f);
        //else if (dist > 1.2944f && dist <= 2.0387f) AddReward(-0.05f);
        //else if (dist <= 1.2944f)                   AddReward(-0.1f);


        numStep += 1;
        //Debug.Log("step : " + numStep);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actionOut = actionsOut.DiscreteActions[0];
        // 왼쪽
        if (Input.GetKey(KeyCode.LeftArrow)) actionOut = 9;

        // 직진
        if (Input.GetKey(KeyCode.UpArrow)) actionOut = 0;

        // 오른쪽
        if (Input.GetKey(KeyCode.RightArrow)) actionOut = 12;
    }

    private void OnCollisionEnter(Collision collision)
    {
        numStep = 0;

        if (collision.collider.CompareTag("wall"))
        {
            AddReward(-30.0f);
            Debug.Log("Detection!");
            EndEpisode();
        }
        if (collision.collider.CompareTag("goal"))
        {
            AddReward(50.0f);
            EndEpisode();
        }
        if (collision.collider.CompareTag("notGoal"))
        {
            AddReward(-30.0f);
            EndEpisode();
        }
    }


}
