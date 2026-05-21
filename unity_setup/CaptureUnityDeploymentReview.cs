// Assets/Editor/CaptureUnityDeploymentReview.cs
// Batch-mode Unity deployment review for Gaussian splat scenes.

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Object = UnityEngine.Object;
using Stopwatch = System.Diagnostics.Stopwatch;

public static class CaptureUnityDeploymentReview
{
    const string DEFAULT_SCENE_PATH = "Assets/Scenes/FactoryGaussian.unity";

    sealed class ViewScore
    {
        public string name = "";
        public string screenshot = "";
        public Vector3 position;
        public Vector3 euler;
        public double renderMs;
        public double avgLuma;
        public double whiteHazeRatio;
        public double brightClipRatio;
        public double darkVoidRatio;
        public double edgeSharpness;
        public bool validImage;
    }

    public static void Run()
    {
        try
        {
            var args = ParseArgs(Environment.GetCommandLineArgs());
            string outputDir = GetArg(args, "-reviewOutputDir", "C:/3d-recon-pipeline/outputs/unity_deployment_review");
            string scenePath = GetArg(args, "-reviewScene", DEFAULT_SCENE_PATH);
            int width = ParseInt(GetArg(args, "-reviewWidth", "1280"), 1280);
            int height = ParseInt(GetArg(args, "-reviewHeight", "720"), 720);

            Directory.CreateDirectory(outputDir);
            Directory.CreateDirectory(Path.Combine(outputDir, "screenshots"));

            var scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            if (!scene.IsValid())
            {
                throw new InvalidOperationException($"Invalid scene: {scenePath}");
            }

            var camera = Object.FindFirstObjectByType<Camera>();
            if (camera == null)
            {
                throw new InvalidOperationException("Scene has no Camera");
            }

            var renderer = Object.FindFirstObjectByType<GaussianSplatRenderer>();
            if (renderer == null)
            {
                throw new InvalidOperationException("Scene has no GaussianSplatRenderer");
            }
            if (renderer.m_Asset == null)
            {
                throw new InvalidOperationException("GaussianSplatRenderer has no asset assigned");
            }

            ConfigureCamera(camera, width, height);

            var target = renderer.transform.position;
            var originalPosition = camera.transform.position;
            var originalRotation = camera.transform.rotation;
            float radius = Vector3.Distance(originalPosition, target);
            if (radius < 0.25f || float.IsNaN(radius))
            {
                radius = 3.0f;
            }

            float heightOffset = originalPosition.y - target.y;
            var views = BuildViews(target, originalPosition, originalRotation, radius, heightOffset);
            var scores = new List<ViewScore>();

            foreach (var view in views)
            {
                camera.transform.position = view.position;
                camera.transform.rotation = view.rotation;
                scores.Add(CaptureView(camera, outputDir, view.name, width, height));
            }

            camera.transform.position = originalPosition;
            camera.transform.rotation = originalRotation;

            string scorePath = Path.Combine(outputDir, "deployment_review_score.json");
            File.WriteAllText(
                scorePath,
                BuildJson(scenePath, outputDir, width, height, renderer, scores),
                new UTF8Encoding(false)
            );

            Debug.Log($"[CaptureUnityDeploymentReview] wrote {scorePath}");
            EditorApplication.Exit(0);
        }
        catch (Exception ex)
        {
            Debug.LogError($"[CaptureUnityDeploymentReview] failed: {ex}");
            EditorApplication.Exit(1);
        }
    }

    static Dictionary<string, string> ParseArgs(string[] argv)
    {
        var result = new Dictionary<string, string>();
        for (int i = 0; i < argv.Length - 1; i++)
        {
            if (argv[i].StartsWith("-", StringComparison.Ordinal))
            {
                result[argv[i]] = argv[i + 1];
            }
        }
        return result;
    }

    static string GetArg(Dictionary<string, string> args, string key, string fallback)
    {
        return args.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value) ? value : fallback;
    }

    static int ParseInt(string value, int fallback)
    {
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) && parsed > 0
            ? parsed
            : fallback;
    }

    static void ConfigureCamera(Camera camera, int width, int height)
    {
        camera.enabled = true;
        camera.aspect = width / (float)Math.Max(height, 1);
        camera.nearClipPlane = Mathf.Max(0.001f, camera.nearClipPlane);
        camera.farClipPlane = Mathf.Max(1000f, camera.farClipPlane);
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = Color.black;
    }

    static List<(string name, Vector3 position, Quaternion rotation)> BuildViews(
        Vector3 target,
        Vector3 originalPosition,
        Quaternion originalRotation,
        float radius,
        float heightOffset)
    {
        var views = new List<(string, Vector3, Quaternion)>
        {
            ("scene_camera", originalPosition, originalRotation),
        };

        Vector3 forward = target - originalPosition;
        forward.y = 0f;
        if (forward.sqrMagnitude < 0.0001f)
        {
            forward = Vector3.forward;
        }

        float baseYaw = Mathf.Atan2(forward.x, forward.z) * Mathf.Rad2Deg;
        float[] yaws = { -60f, -30f, 0f, 30f, 60f, 120f, -120f };
        foreach (float yawOffset in yaws)
        {
            float yaw = baseYaw + yawOffset;
            float rad = yaw * Mathf.Deg2Rad;
            var pos = target - new Vector3(Mathf.Sin(rad), 0f, Mathf.Cos(rad)) * radius;
            pos.y = target.y + heightOffset;
            var rot = Quaternion.LookRotation(target - pos, Vector3.up);
            views.Add(($"orbit_{yawOffset:+0;-0;0}", pos, rot));
        }

        return views;
    }

    static ViewScore CaptureView(Camera camera, string outputDir, string viewName, int width, int height)
    {
        var score = new ViewScore();
        score.name = viewName;
        score.position = camera.transform.position;
        score.euler = camera.transform.eulerAngles;

        var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
        var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        var previousTarget = camera.targetTexture;
        var previousActive = RenderTexture.active;

        try
        {
            camera.targetTexture = rt;
            RenderTexture.active = rt;
            var sw = Stopwatch.StartNew();
            camera.Render();
            sw.Stop();
            tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            tex.Apply(false);
            score.renderMs = sw.Elapsed.TotalMilliseconds;

            AnalyzePixels(tex, score);

            string relativePath = $"screenshots/{viewName}.png";
            string absolutePath = Path.Combine(outputDir, relativePath);
            File.WriteAllBytes(absolutePath, tex.EncodeToPNG());
            score.screenshot = relativePath.Replace("\\", "/");
        }
        finally
        {
            camera.targetTexture = previousTarget;
            RenderTexture.active = previousActive;
            Object.DestroyImmediate(tex);
            Object.DestroyImmediate(rt);
        }

        return score;
    }

    static void AnalyzePixels(Texture2D tex, ViewScore score)
    {
        var pixels = tex.GetPixels32();
        if (pixels.Length == 0)
        {
            return;
        }

        double lumaSum = 0.0;
        int whiteHaze = 0;
        int brightClip = 0;
        int darkVoid = 0;
        int valid = 0;

        foreach (var px in pixels)
        {
            double r = px.r / 255.0;
            double g = px.g / 255.0;
            double b = px.b / 255.0;
            double max = Math.Max(r, Math.Max(g, b));
            double min = Math.Min(r, Math.Min(g, b));
            double luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            double saturation = max <= 0.00001 ? 0.0 : (max - min) / max;

            lumaSum += luma;
            if (luma > 0.88 && saturation < 0.22)
            {
                whiteHaze++;
            }
            if (r > 0.985 || g > 0.985 || b > 0.985)
            {
                brightClip++;
            }
            if (luma < 0.015)
            {
                darkVoid++;
            }
            if (luma > 0.015 && luma < 0.985)
            {
                valid++;
            }
        }

        score.avgLuma = lumaSum / pixels.Length;
        score.whiteHazeRatio = whiteHaze / (double)pixels.Length;
        score.brightClipRatio = brightClip / (double)pixels.Length;
        score.darkVoidRatio = darkVoid / (double)pixels.Length;
        score.validImage = valid > pixels.Length * 0.08;
        score.edgeSharpness = ComputeEdgeSharpness(pixels, tex.width, tex.height);
    }

    static double ComputeEdgeSharpness(Color32[] pixels, int width, int height)
    {
        if (width < 3 || height < 3)
        {
            return 0.0;
        }

        double sum = 0.0;
        int count = 0;
        for (int y = 1; y < height - 1; y += 2)
        {
            int row = y * width;
            for (int x = 1; x < width - 1; x += 2)
            {
                double left = Luma(pixels[row + x - 1]);
                double right = Luma(pixels[row + x + 1]);
                double down = Luma(pixels[row - width + x]);
                double up = Luma(pixels[row + width + x]);
                sum += Math.Abs(right - left) + Math.Abs(up - down);
                count++;
            }
        }
        return count == 0 ? 0.0 : sum / count;
    }

    static double Luma(Color32 px)
    {
        return (0.2126 * px.r + 0.7152 * px.g + 0.0722 * px.b) / 255.0;
    }

    static string BuildJson(
        string scenePath,
        string outputDir,
        int width,
        int height,
        GaussianSplatRenderer renderer,
        List<ViewScore> scores)
    {
        int validViews = 0;
        double haze = 0.0;
        double clip = 0.0;
        double dark = 0.0;
        double sharp = 0.0;
        double renderMs = 0.0;
        foreach (var score in scores)
        {
            if (score.validImage)
            {
                validViews++;
            }
            haze += score.whiteHazeRatio;
            clip += score.brightClipRatio;
            dark += score.darkVoidRatio;
            sharp += score.edgeSharpness;
            renderMs += score.renderMs;
        }

        double n = Math.Max(scores.Count, 1);
        double meanHaze = haze / n;
        double meanClip = clip / n;
        double meanDark = dark / n;
        double meanSharp = sharp / n;
        double meanRenderMs = renderMs / n;

        bool importSuccess = renderer != null && renderer.m_Asset != null;
        bool reviewPass = importSuccess
            && validViews >= Math.Min(5, scores.Count)
            && meanHaze <= 0.18
            && meanClip <= 0.28
            && meanDark <= 0.55
            && meanSharp >= 0.018;

        var sb = new StringBuilder();
        sb.AppendLine("{");
        AddJson(sb, "schema_version", "1", comma: true, raw: true, indent: 2);
        AddJson(sb, "review_type", "unity_deployment_auto_review", comma: true, indent: 2);
        AddJson(sb, "recorded_at", DateTimeOffset.Now.ToString("o"), comma: true, indent: 2);
        AddJson(sb, "scene_path", scenePath, comma: true, indent: 2);
        AddJson(sb, "output_dir", outputDir.Replace("\\", "/"), comma: true, indent: 2);
        AddJson(sb, "resolution", $"{{\"width\":{width},\"height\":{height}}}", comma: true, raw: true, indent: 2);
        AddJson(sb, "import_success", importSuccess ? "true" : "false", comma: true, raw: true, indent: 2);
        AddJson(sb, "asset_name", importSuccess ? renderer.m_Asset.name : "", comma: true, indent: 2);
        AddJson(sb, "splat_count", importSuccess ? renderer.m_Asset.splatCount.ToString(CultureInfo.InvariantCulture) : "0", comma: true, raw: true, indent: 2);
        sb.AppendLine("  \"metrics\": {");
        AddJson(sb, "views_evaluated", scores.Count.ToString(CultureInfo.InvariantCulture), comma: true, raw: true, indent: 4);
        AddJson(sb, "valid_views", validViews.ToString(CultureInfo.InvariantCulture), comma: true, raw: true, indent: 4);
        AddJson(sb, "render_ms_mean", F(meanRenderMs), comma: true, raw: true, indent: 4);
        AddJson(sb, "white_haze_ratio_mean", F(meanHaze), comma: true, raw: true, indent: 4);
        AddJson(sb, "bright_clip_ratio_mean", F(meanClip), comma: true, raw: true, indent: 4);
        AddJson(sb, "dark_void_ratio_mean", F(meanDark), comma: true, raw: true, indent: 4);
        AddJson(sb, "edge_sharpness_mean", F(meanSharp), comma: false, raw: true, indent: 4);
        sb.AppendLine("  },");
        sb.AppendLine("  \"thresholds\": {");
        AddJson(sb, "white_haze_ratio_mean_max", "0.18", comma: true, raw: true, indent: 4);
        AddJson(sb, "bright_clip_ratio_mean_max", "0.28", comma: true, raw: true, indent: 4);
        AddJson(sb, "dark_void_ratio_mean_max", "0.55", comma: true, raw: true, indent: 4);
        AddJson(sb, "edge_sharpness_mean_min", "0.018", comma: false, raw: true, indent: 4);
        sb.AppendLine("  },");
        AddJson(sb, "deployment_review_pass", reviewPass ? "true" : "false", comma: true, raw: true, indent: 2);
        AddJson(sb, "deployable_pass", "false", comma: true, raw: true, indent: 2);
        AddJson(sb, "human_review_required", "true", comma: true, raw: true, indent: 2);
        sb.AppendLine("  \"failure_tags\": [");
        var tags = BuildFailureTags(importSuccess, validViews, scores.Count, meanHaze, meanClip, meanDark, meanSharp);
        for (int i = 0; i < tags.Count; i++)
        {
            sb.Append("    \"").Append(Escape(tags[i])).Append("\"");
            sb.AppendLine(i == tags.Count - 1 ? "" : ",");
        }
        sb.AppendLine("  ],");
        sb.AppendLine("  \"views\": [");
        for (int i = 0; i < scores.Count; i++)
        {
            var score = scores[i];
            sb.AppendLine("    {");
            AddJson(sb, "name", score.name, comma: true, indent: 6);
            AddJson(sb, "screenshot", score.screenshot, comma: true, indent: 6);
            AddJson(sb, "position", Vec(score.position), comma: true, raw: true, indent: 6);
            AddJson(sb, "euler", Vec(score.euler), comma: true, raw: true, indent: 6);
            AddJson(sb, "render_ms", F(score.renderMs), comma: true, raw: true, indent: 6);
            AddJson(sb, "avg_luma", F(score.avgLuma), comma: true, raw: true, indent: 6);
            AddJson(sb, "white_haze_ratio", F(score.whiteHazeRatio), comma: true, raw: true, indent: 6);
            AddJson(sb, "bright_clip_ratio", F(score.brightClipRatio), comma: true, raw: true, indent: 6);
            AddJson(sb, "dark_void_ratio", F(score.darkVoidRatio), comma: true, raw: true, indent: 6);
            AddJson(sb, "edge_sharpness", F(score.edgeSharpness), comma: true, raw: true, indent: 6);
            AddJson(sb, "valid_image", score.validImage ? "true" : "false", comma: false, raw: true, indent: 6);
            sb.Append("    }").AppendLine(i == scores.Count - 1 ? "" : ",");
        }
        sb.AppendLine("  ]");
        sb.AppendLine("}");
        return sb.ToString();
    }

    static List<string> BuildFailureTags(bool importSuccess, int validViews, int totalViews, double haze, double clip, double dark, double sharp)
    {
        var tags = new List<string>();
        if (!importSuccess) tags.Add("import_failed");
        if (validViews < Math.Min(5, totalViews)) tags.Add("invalid_or_black_views");
        if (haze > 0.18) tags.Add("white_haze");
        if (clip > 0.28) tags.Add("highlight_clip");
        if (dark > 0.55) tags.Add("dark_void");
        if (sharp < 0.018) tags.Add("low_edge_sharpness");
        if (tags.Count == 0) tags.Add("none");
        return tags;
    }

    static void AddJson(StringBuilder sb, string key, string value, bool comma, int indent, bool raw = false)
    {
        sb.Append(' ', indent);
        sb.Append('"').Append(Escape(key)).Append("\": ");
        if (raw)
        {
            sb.Append(value);
        }
        else
        {
            sb.Append('"').Append(Escape(value)).Append('"');
        }
        if (comma)
        {
            sb.Append(',');
        }
        sb.AppendLine();
    }

    static string Vec(Vector3 value)
    {
        return $"{{\"x\":{F(value.x)},\"y\":{F(value.y)},\"z\":{F(value.z)}}}";
    }

    static string F(double value)
    {
        return value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    static string Escape(string value)
    {
        return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
#endif
