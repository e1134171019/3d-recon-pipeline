#if UNITY_EDITOR
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class BatchCreateScene
{
    const string ASSET_PATH = "Assets/GaussianAssets/point_cloud_unity.asset";
    const string SCENE_PATH = "Assets/Scenes/FactoryGaussian.unity";

    [MenuItem("FactoryScene/1. 建立或重建 FactoryGaussian 場景")]
    public static void Run()
    {
        Debug.Log("[BatchCreateScene] 開始建立場景...");

        var splat = AssetDatabase.LoadAssetAtPath<GaussianSplatAsset>(ASSET_PATH);
        if (splat == null)
        {
            Debug.LogError($"[BatchCreateScene] 找不到 GaussianSplatAsset：{ASSET_PATH}");
            EditorApplication.Exit(1);
            return;
        }

        if (!AssetDatabase.IsValidFolder("Assets/Scenes"))
        {
            AssetDatabase.CreateFolder("Assets", "Scenes");
        }

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        var camGO = new GameObject("Main Camera");
        camGO.tag = "MainCamera";
        var cam = camGO.AddComponent<Camera>();
        cam.transform.position = new Vector3(0f, 1f, -3f);
        cam.transform.rotation = Quaternion.Euler(10f, 0f, 0f);
        cam.fieldOfView = 60f;
        cam.nearClipPlane = 0.01f;
        cam.farClipPlane = 1000f;
        if (cam.GetComponent<OrbitCamera>() == null)
        {
            cam.gameObject.AddComponent<OrbitCamera>();
        }

        var lightGO = new GameObject("Directional Light");
        var light = lightGO.AddComponent<Light>();
        light.type = LightType.Directional;
        lightGO.SetActive(false);

        var splatGO = new GameObject("FactorySplat");
        var renderer = splatGO.AddComponent<GaussianSplatRenderer>();
        renderer.m_Asset = splat;

        EditorSceneManager.SaveScene(scene, SCENE_PATH);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log($"[BatchCreateScene] 完成：{SCENE_PATH}");
        EditorApplication.Exit(0);
    }
}
#endif
