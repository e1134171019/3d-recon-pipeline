#if UNITY_EDITOR
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class FactorySceneTools
{
    [MenuItem("FactoryScene/2. 套用所選 Gaussian Asset 並重設視角")]
    public static void ApplySelectedGaussianAssetAndReframe()
    {
        var selected = Selection.activeObject as GaussianSplatAsset;
        if (selected == null)
        {
            Debug.LogWarning("[FactorySceneTools] 請先在 Project 視窗選一顆 GaussianSplatAsset。");
            return;
        }

        var renderer = Object.FindFirstObjectByType<GaussianSplatRenderer>();
        if (renderer == null)
        {
            Debug.LogWarning("[FactorySceneTools] 找不到 FactorySplat / GaussianSplatRenderer。");
            return;
        }

        renderer.m_Asset = selected;

        var dirLight = GameObject.Find("Directional Light");
        if (dirLight != null)
        {
            dirLight.SetActive(false);
            EditorUtility.SetDirty(dirLight);
        }

        EditorUtility.SetDirty(renderer);
        EditorSceneManager.MarkAllScenesDirty();
        ReframeCamera.Run();

        Debug.Log($"[FactorySceneTools] 已套用資產：{selected.name}");
    }
}
#endif
