using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

[RequireComponent(typeof(Camera))]
public class OrbitCamera : MonoBehaviour
{
    public Vector3 target = Vector3.zero;
    public float orbitSpeed = 200f;
    public float panSpeed = 0.003f;
    public float zoomSpeed = 0.5f;
    public float minDistance = 0.3f;
    public float maxDistance = 30f;

    float _yaw;
    float _pitch;
    float _distance;
    float _initYaw;
    float _initPitch;
    float _initDist;
    Vector3 _initTarget;
    float _lastClickTime = -1f;

    void Start()
    {
        var splat = FindFirstObjectByType<GaussianSplatting.Runtime.GaussianSplatRenderer>();
        if (splat != null)
        {
            target = splat.transform.position;
        }

        Vector3 offset = transform.position - target;
        _distance = Mathf.Clamp(offset.magnitude, minDistance, maxDistance);
        if (_distance < 0.01f)
        {
            _distance = 5f;
        }

        Vector3 dir = offset.normalized;
        _pitch = Mathf.Asin(Mathf.Clamp(dir.y, -1f, 1f)) * Mathf.Rad2Deg;
        _yaw = Mathf.Atan2(dir.x, dir.z) * Mathf.Rad2Deg;

        _initYaw = _yaw;
        _initPitch = _pitch;
        _initDist = _distance;
        _initTarget = target;
    }

    void LateUpdate()
    {
        float dt = Time.unscaledDeltaTime;
        float dx = GetMouseDelta().x;
        float dy = GetMouseDelta().y;
        bool leftHeld = IsLeftMouseHeld();
        bool rightHeld = IsRightMouseHeld();
        bool leftPressed = WasLeftMousePressedThisFrame();
        float scroll = GetScrollDelta();

        if (leftHeld)
        {
            _yaw += dx * orbitSpeed * dt;
            _pitch -= dy * orbitSpeed * dt;
            _pitch = Mathf.Clamp(_pitch, -80f, 80f);
        }

        if (rightHeld)
        {
            float s = _distance * panSpeed * dt;
            target -= transform.right * dx * s;
            target -= transform.up * dy * s;
        }

        if (Mathf.Abs(scroll) > 0.001f)
        {
            _distance = Mathf.Clamp(_distance - scroll * zoomSpeed * _distance, minDistance, maxDistance);
        }

        if (leftPressed)
        {
            if (Time.unscaledTime - _lastClickTime < 0.3f)
            {
                _yaw = _initYaw;
                _pitch = _initPitch;
                _distance = _initDist;
                target = _initTarget;
            }
            _lastClickTime = Time.unscaledTime;
        }

        Quaternion rot = Quaternion.Euler(_pitch, _yaw, 0f);
        transform.position = target + rot * new Vector3(0f, 0f, -_distance);
        transform.LookAt(target, Vector3.up);
    }

    Vector2 GetMouseDelta()
    {
#if ENABLE_INPUT_SYSTEM
        if (Mouse.current != null)
        {
            return Mouse.current.delta.ReadValue();
        }
#endif
        return new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
    }

    float GetScrollDelta()
    {
#if ENABLE_INPUT_SYSTEM
        if (Mouse.current != null)
        {
            return Mouse.current.scroll.ReadValue().y * 0.01f;
        }
#endif
        return Input.GetAxis("Mouse ScrollWheel");
    }

    bool IsLeftMouseHeld()
    {
#if ENABLE_INPUT_SYSTEM
        if (Mouse.current != null)
        {
            return Mouse.current.leftButton.isPressed;
        }
#endif
        return Input.GetMouseButton(0);
    }

    bool IsRightMouseHeld()
    {
#if ENABLE_INPUT_SYSTEM
        if (Mouse.current != null)
        {
            return Mouse.current.rightButton.isPressed;
        }
#endif
        return Input.GetMouseButton(1);
    }

    bool WasLeftMousePressedThisFrame()
    {
#if ENABLE_INPUT_SYSTEM
        if (Mouse.current != null)
        {
            return Mouse.current.leftButton.wasPressedThisFrame;
        }
#endif
        return Input.GetMouseButtonDown(0);
    }
}
