// Assets/Editor/GaussianSplatBatchImport.cs
// 可在 Unity GUI 或 batch mode 下重新匯入最新的 point_cloud_unity.ply

#if UNITY_EDITOR
using System;
using System.IO;
using System.Reflection;
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class GaussianSplatBatchImport
{
    const string INPUT_FILE = "Assets/GaussianSplats/point_cloud_unity.ply";
    const string INPUT_FILE_PRUNED = "Assets/GaussianSplats/point_cloud_unity_pruned.ply";
    const string INPUT_FILE_A_PRUNED = "Assets/GaussianSplats/point_cloud_unity_A_pruned.ply";
    const string OUTPUT_FOLDER = "Assets/GaussianAssets";
    const string SCENE_PATH = "Assets/Scenes/FactoryGaussian.unity";
    const string SCENE_PATH_A = "Assets/Scenes/FactoryGaussian_A.unity";

    [MenuItem("FactoryScene/4. Reimport point_cloud_unity.asset")]
    public static void ImportLatestPointCloudUnityMenu()
    {
        ImportPointCloudUnity(INPUT_FILE, false);
    }

    public static void ImportLatestPointCloudUnity()
    {
        ImportPointCloudUnity(INPUT_FILE, true);
    }

    [MenuItem("FactoryScene/5. Reimport pruned point_cloud_unity.asset")]
    public static void ImportPrunedPointCloudUnityMenu()
    {
        ImportPointCloudUnity(INPUT_FILE_PRUNED, false);
    }

    public static void ImportPrunedPointCloudUnity()
    {
        ImportPointCloudUnity(INPUT_FILE_PRUNED, SCENE_PATH, true);
    }

    [MenuItem("FactoryScene/6. Reimport A/B pruned point_cloud_unity_A.asset")]
    public static void ImportPrunedPointCloudUnityAMenu()
    {
        ImportPointCloudUnity(INPUT_FILE_A_PRUNED, SCENE_PATH_A, false);
    }

    public static void ImportPrunedPointCloudUnityA()
    {
        ImportPointCloudUnity(INPUT_FILE_A_PRUNED, SCENE_PATH_A, true);
    }

    static void ImportPointCloudUnity(string inputFile, bool exitWhenDone)
    {
        ImportPointCloudUnity(inputFile, SCENE_PATH, exitWhenDone);
    }

    static void ImportPointCloudUnity(string inputFile, string scenePath, bool exitWhenDone)
    {
        try
        {
            if (AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(inputFile) == null)
                throw new InvalidOperationException($"找不到輸入檔：{inputFile}");

            var creatorType = Type.GetType(
                "GaussianSplatting.Editor.GaussianSplatAssetCreator, GaussianSplattingEditor");
            if (creatorType == null)
                throw new InvalidOperationException("找不到 GaussianSplatAssetCreator，請確認插件已安裝且已完成編譯");

            var creator = ScriptableObject.CreateInstance(creatorType);
            try
            {
                SetField(creatorType, creator, "m_InputFile", inputFile);
                SetField(creatorType, creator, "m_OutputFolder", OUTPUT_FOLDER);
                SetField(creatorType, creator, "m_ImportCameras", false);

                var createMethod = creatorType.GetMethod("CreateAsset", BindingFlags.Instance | BindingFlags.NonPublic);
                if (createMethod == null)
                    throw new MissingMethodException("GaussianSplatAssetCreator.CreateAsset");

                createMethod.Invoke(creator, null);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                var outputAsset = $"{OUTPUT_FOLDER}/{Path.GetFileNameWithoutExtension(inputFile)}.asset";
                var asset = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(outputAsset);
                if (asset == null)
                    throw new InvalidOperationException($"匯入後找不到輸出資產：{outputAsset}");

                BindAssetToScene(outputAsset, scenePath);

                Debug.Log($"[GaussianSplatBatchImport] 匯入完成：{outputAsset}");
                Selection.activeObject = asset;

                if (exitWhenDone)
                    EditorApplication.Exit(0);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(creator);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[GaussianSplatBatchImport] 匯入失敗：{ex}");
            if (exitWhenDone)
                EditorApplication.Exit(1);
        }
    }

    static void BindAssetToScene(string assetPath, string scenePath)
    {
        var workingScenePath = scenePath;
        if (scenePath != SCENE_PATH && !AssetDatabase.LoadAssetAtPath<SceneAsset>(scenePath))
        {
            if (!AssetDatabase.CopyAsset(SCENE_PATH, scenePath))
                throw new InvalidOperationException($"無法建立 A/B 場景副本：{scenePath}");
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        var scene = EditorSceneManager.OpenScene(workingScenePath, OpenSceneMode.Single);
        var renderer = UnityEngine.Object.FindFirstObjectByType<GaussianSplatRenderer>();
        if (renderer == null)
            throw new InvalidOperationException($"場景中找不到 GaussianSplatRenderer：{workingScenePath}");

        var asset = AssetDatabase.LoadAssetAtPath<GaussianSplatAsset>(assetPath);
        if (asset == null)
            throw new InvalidOperationException($"找不到 GaussianSplatAsset：{assetPath}");

        Undo.RecordObject(renderer, "Bind Gaussian Splat Asset");
        renderer.m_Asset = asset;
        EditorUtility.SetDirty(renderer);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }

    static void SetField(Type type, object instance, string name, object value)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field == null)
            throw new MissingFieldException(type.FullName, name);
        field.SetValue(instance, value);
    }
}
#endif
