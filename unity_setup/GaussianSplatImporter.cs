// Assets/Editor/GaussianSplatImporter.cs
// 自動從 Assets/GaussianSplats/ 匯入 .ply 並建立場景
// 使用：Unity 選單 → FactoryScene → Import Gaussian Splat

#if UNITY_EDITOR
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;

public class GaussianSplatImporter : Editor
{
    const string SPLAT_FOLDER  = "Assets/GaussianSplats";
    const string SCENE_PATH    = "Assets/Scenes/FactoryGaussian.unity";
    const string OUTPUT_FOLDER = "Assets/GaussianAssets";
    const string PIPELINE_PLY_DIR = "C:/3d-recon-pipeline/outputs/3DGS_models/ply";

    [MenuItem("FactoryScene/0. Sync Latest Pipeline PLY")]
    public static void SyncLatestPipelinePly()
    {
        if (!AssetDatabase.IsValidFolder("Assets/GaussianSplats"))
        {
            AssetDatabase.CreateFolder("Assets", "GaussianSplats");
        }

        var pipelineDir = Path.GetFullPath(PIPELINE_PLY_DIR);
        if (!Directory.Exists(pipelineDir))
        {
            EditorUtility.DisplayDialog(
                "找不到 pipeline 輸出",
                $"找不到目錄：{pipelineDir}",
                "OK");
            return;
        }

        var srcFiles = new[]
        {
            Path.Combine(pipelineDir, "point_cloud_final.ply"),
            Path.Combine(pipelineDir, "point_cloud_unity.ply"),
        };

        int copied = 0;
        foreach (var src in srcFiles)
        {
            if (!File.Exists(src))
                continue;

            var dst = Path.Combine(Path.GetFullPath(SPLAT_FOLDER), Path.GetFileName(src));
            File.Copy(src, dst, true);
            copied += 1;
            Debug.Log($"[GaussianSplatImporter] Copied: {src} -> {dst}");
        }

        AssetDatabase.Refresh();

        if (copied == 0)
        {
            EditorUtility.DisplayDialog(
                "沒有可同步的 PLY",
                $"在 {pipelineDir} 下沒有找到 point_cloud_final.ply 或 point_cloud_unity.ply",
                "OK");
            return;
        }

        EditorUtility.DisplayDialog(
            "同步完成",
            $"已同步 {copied} 個 PLY 到 {SPLAT_FOLDER}\n\n" +
            "下一步請執行：FactoryScene → 1. Import Gaussian Splat PLY",
            "OK");
    }

    [MenuItem("FactoryScene/1. Import Gaussian Splat PLY")]
    public static void ImportSplat()
    {
        // 確保輸出目錄存在
        if (!AssetDatabase.IsValidFolder(OUTPUT_FOLDER))
        {
            var parent = Path.GetDirectoryName(OUTPUT_FOLDER).Replace("\\", "/");
            var folderName = Path.GetFileName(OUTPUT_FOLDER);
            AssetDatabase.CreateFolder(parent, folderName);
        }

        // 找所有 .ply 檔案
        var plyGuids = AssetDatabase.FindAssets("t:DefaultAsset", new[] { SPLAT_FOLDER })
            .Select(g => AssetDatabase.GUIDToAssetPath(g))
            .Where(p => p.EndsWith(".ply"))
            .ToArray();

        if (plyGuids.Length == 0)
        {
            EditorUtility.DisplayDialog(
                "找不到 PLY 檔案",
                $"請確認 {SPLAT_FOLDER} 下有 .ply 檔案\n\n" +
                "訓練輸出路徑：C:\\3d-recon-pipeline\\outputs\\3DGS_models\\ply\\",
                "OK");
            return;
        }

        Debug.Log($"[GaussianSplatImporter] 找到 {plyGuids.Length} 個 .ply 檔案");
        foreach (var p in plyGuids)
            Debug.Log($"  → {p}");

        // 嘗試呼叫 aras-p Gaussian Splatting 插件的匯入功能
        // 插件類型：GaussianSplatAssetCreator（需已安裝插件）
        var creatorType = System.Type.GetType(
            "GaussianSplatting.Editor.GaussianSplatAssetCreator, GaussianSplattingEditor");

        if (creatorType != null)
        {
            // 插件已安裝：開啟 Asset Creator 視窗
            EditorWindow.GetWindow(creatorType, false, "Gaussian Splat Creator").Show();
            EditorUtility.DisplayDialog(
                "Gaussian Splat Creator 已開啟",
                $"請在 Creator 視窗中：\n" +
                $"1. Input PLY File → 選擇 {SPLAT_FOLDER} 下的 .ply\n" +
                $"2. Output Folder → {OUTPUT_FOLDER}\n" +
                $"3. 按 Create Asset",
                "OK");
        }
        else
        {
            // 插件未安裝：顯示 .ply 路徑讓用戶手動操作
            var paths = string.Join("\n", plyGuids);
            EditorUtility.DisplayDialog(
                "請安裝 Gaussian Splatting 插件",
                "插件未偵測到。請確認 Packages/manifest.json 包含：\n\n" +
                "\"org.nesnausk.gaussian-splatting\": \"https://github.com/aras-p/UnityGaussianSplatting.git?path=package\"\n\n" +
                $"找到的 .ply 檔案：\n{paths}",
                "OK");
        }
    }

    [MenuItem("FactoryScene/2. Create Factory Scene")]
    public static void CreateFactoryScene()
    {
        // 建立 Scenes 目錄
        if (!AssetDatabase.IsValidFolder("Assets/Scenes"))
            AssetDatabase.CreateFolder("Assets", "Scenes");

        // 建立新場景
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // 建立主相機（對應 COLMAP 相機參數）
        var cam = Camera.main ?? new GameObject("Main Camera").AddComponent<Camera>();
        cam.transform.position = new Vector3(0, 1.5f, -3f);
        cam.transform.LookAt(Vector3.zero);
        cam.fieldOfView = 60f;

        // 建立 GaussianSplatRenderer GameObject（需插件）
        var splatGO = new GameObject("FactorySplat");
        splatGO.transform.position = Vector3.zero;

        // 若插件存在，加上 GaussianSplatRenderer 元件
        var rendererType = System.Type.GetType(
            "GaussianSplatting.Runtime.GaussianSplatRenderer, GaussianSplatting");
        if (rendererType != null)
        {
            splatGO.AddComponent(rendererType);
            Debug.Log("[GaussianSplatImporter] GaussianSplatRenderer 已加入 FactorySplat");
        }
        else
        {
            Debug.LogWarning("[GaussianSplatImporter] 未找到 GaussianSplatRenderer，請確認插件已安裝");
        }

        // 儲存場景
        EditorSceneManager.SaveScene(scene, SCENE_PATH);
        AssetDatabase.Refresh();

        Debug.Log($"[GaussianSplatImporter] 場景已儲存：{SCENE_PATH}");
        EditorUtility.DisplayDialog(
            "場景建立完成",
            $"場景路徑：{SCENE_PATH}\n\n" +
            "請在 Hierarchy 中選擇 FactorySplat，\n" +
            "將匯入的 GaussianSplatAsset 拖入 Asset 欄位。",
            "OK");
    }

    [MenuItem("FactoryScene/3. Show Output Files")]
    public static void ShowOutputFiles()
    {
        var absPath = Path.GetFullPath(PIPELINE_PLY_DIR);
        if (Directory.Exists(absPath))
        {
            var files = Directory.GetFiles(absPath, "*", SearchOption.AllDirectories);
            Debug.Log($"=== outputs/3DGS_models/ply 輸出檔案（共 {files.Length} 個）===");
            foreach (var f in files.Take(30))
                Debug.Log($"  {Path.GetRelativePath(absPath, f)}  ({new FileInfo(f).Length / 1024} KB)");
        }
        else
        {
            Debug.LogError($"找不到目錄：{absPath}");
        }
        EditorUtility.RevealInFinder(absPath);
    }
}
#endif
