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
       
        // �������� �ʱ�ȭ
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        //tr.localPosition = startPosition;
        //tr.localEulerAngles=startRotation;
        // ��ġ �ʱ�ȭ
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        // �������

        //transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

        //Debug.Log("position : " + tr.position);
        //Debug.Log("Rotation : " + tr.eulerAngles);
        //Debug.Log(" position : " + transform.position);
        //Debug.Log("red.. scale : " + transform.localScale);
        //Debug.Log("box collider center : " + GetComponentInChildren<BoxCollider>().center);
        //Debug.Log("box collider size : " + GetComponentInChildren<BoxCollider>().size);
        
        // ������� �ʿ���� ��
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

        // action ���� �� agent ��ġ
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
        // action ����
        transform.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward);
        transform.Rotate(new Vector3(0f, Turn, 0f));


        // new Vector3(0.8575f/2, 0.5201f/2, 2.2634f/2)

        // ���⼭����

        // Physics.OverlapBox(������Ʈ �߽�, ������Ʈ box collider ũ���� ����)
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

        // ������� �ʿ���� �κ�.. Physics.OverlapBox ���� OnCollisionEnter ���

        // action ���� �� agent �̵��� �Ÿ�
        // ���� �ӵ� 5.0f  -> �� 0.1
        // ���� �ӵ� 7.5f  -> �� 0.15
        // ���� �ӵ� 10.0f -> �� 0.2
        distanceTraveled = Vector3.Distance(transform.position, currentPosition);
        //Debug.Log("distance traveled : " + distanceTraveled);

        // action ���� �� �̵��� �Ÿ��� ���� ���� �ο�
        // ������ �� 0.2 �̵����� ���(���� �ӵ� 10.0f) ���� �ο�
        if (distanceTraveled > 0.17) AddReward(0.5f);


        // ���� ����
        float frontDist = 0;
        float leftDist = 0;
        float rightDist = 0;
        //float dist = 0;
        // ���� : rayIndex = 0
        // ������ : rayIndex = 1
        // ���� : rayIndex = 2
        // Ray Length = 4
        // �¿� ray length = 4 * 1.5 = 6
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

        // action ���� �� ���� ������ ���� ���� ����
        // ���� ����
        if (rightDist > 0.6864f && rightDist <= 1.2864f) AddReward(0.6f);
        // ���� ���� �� ����
        else if (rightDist > 1.2864f && rightDist <= 2.9264f) AddReward(0.1f);
        // ���� ���
        // else if (rightDist > 1.2864f && rightDist <= 3.3764f) AddReward(0.2f);
        // �߾Ӽ� ħ��
        else if (rightDist > 3.3764f) AddReward(-1f);

        //dist = leftDist - rightDist;
        ////Debug.Log("Ray distance : " + dist);
        //// �� ����� ��
        //if (dist > 4.8084f)                         AddReward(-0.04f);
        //else if (dist > 3.2392f && dist <= 4.8084f) AddReward(-0.02f);
        //else if (dist > 2.7203f && dist <= 3.2392f) AddReward(-0.01f);
        //// ���� ����
        //else if (dist > 2.2075f && dist <= 2.7203f) AddReward(0.005f);
        //// �߾Ӽ� ����� ��
        //else if (dist > 2.0387f && dist <= 2.2075f) AddReward(-0.03f);
        //else if (dist > 1.2944f && dist <= 2.0387f) AddReward(-0.05f);
        //else if (dist <= 1.2944f)                   AddReward(-0.1f);


        numStep += 1;
        //Debug.Log("step : " + numStep);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actionOut = actionsOut.DiscreteActions[0];
        // ����
        if (Input.GetKey(KeyCode.LeftArrow)) actionOut = 9;

        // ����
        if (Input.GetKey(KeyCode.UpArrow)) actionOut = 0;

        // ������
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
