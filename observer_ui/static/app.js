const { createApp } = Vue;

const ARCHITECTURE_ABBREVIATIONS = {
  production_system: "PS",
  l0_pipeline: "L0",
  video_extract: "VE",
  image_source_gate: "ISG",
  sfm_pipeline: "SFM",
  feature_gate: "FG",
  sparse_model: "SM",
  training_pipeline: "3DG",
  mcmc_runtime: "MC",
  training_metrics: "TM",
  unity_pipeline: "UNI",
  ply_export: "PLY",
  unity_import: "IMP",
  deployment_review: "DR",
  reference_gate: "RG",
  observer_system: "OBS",
  observer_backend: "OBE",
  observer_vue: "VUE",
  decision_system: "DS",
  formal_decision_runtime: "FDR",
  contract_intake: "CI",
  candidate_pool: "CP",
  arbiter: "ARB",
  shared_decision: "SD",
  map_strategy_pack: "MSP",
  pointcloud_validator: "PCV",
  map_validator: "MV",
  production_param_gate: "PPG",
  feedback_audit: "FA",
  current_state: "CS",
  outcome_feedback: "OF",
  learning_curve: "LC",
  offline_learning: "OL",
  ollama_teacher: "OT",
  historical_backfill: "HB",
  pytorch_learner: "PL",
  governance_system: "GOV",
  formal_docs: "FD",
  experiment_history: "EH",
  gate_policy: "GP",
  resources_system: "RES",
  video_frames: "VF",
  colmap_resource: "COL",
  gpu_cuda: "GPU",
  unity_bendviewer: "UBV",
  ollama_resource: "OR",
};

const ARCHITECTURE_LOGIC_LABELS = {
  production_system: ["把影像變成可部署地圖", "生產主線 · 從原始素材一路交付到 Unity"],
  l0_pipeline: ["準備可信任的影像來源", "輸入閘門 · 先阻擋模糊、曝光與壓縮污染"],
  video_extract: ["拆出原始影格", "輸入準備 · 建立不二次壓縮的 L0 母資料"],
  image_source_gate: ["判斷影像是否值得訓練", "品質閘門 · 比較格式、解析度與可用畫面比例"],
  sfm_pipeline: ["建立相機位置與場景骨架", "幾何階段 · 驗證視角能否形成可靠空間"],
  feature_gate: ["檢查影像彼此是否對得上", "幾何閘門 · 阻擋特徵不足與錯誤匹配"],
  sparse_model: ["保存相機軌跡與稀疏幾何", "正式證據 · 提供訓練與同視角驗證基準"],
  training_pipeline: ["把幾何訓練成高斯地圖", "正式訓練 · 以 MCMC 建立可渲染場景"],
  mcmc_runtime: ["執行目前正式訓練策略", "主線能力 · 維持最佳已驗證參數組合"],
  training_metrics: ["量測離線重建品質", "品質證據 · 比較 PSNR、SSIM 與 LPIPS"],
  unity_pipeline: ["把高斯地圖送進 Unity", "部署鏈 · 驗證轉換後是否仍能正確呈現"],
  ply_export: ["轉成 Unity 可讀的高斯檔", "格式轉換 · 保留座標、旋轉、顏色與透明度"],
  unity_import: ["把高斯資產綁進場景", "場景建立 · 匯入資產並接上正式檢視場景"],
  deployment_review: ["檢查 Unity 是否明顯壞掉", "部署閘門 · 多視角檢查霧化、過曝與破損"],
  reference_gate: ["比較 Unity 與真實參考影像", "核心缺口 · 判斷渲染品質是否真的可交付"],
  observer_system: ["呈現整個專案正在做什麼", "唯讀觀測 · 顯示證據、狀態與活動路徑"],
  observer_backend: ["彙整正式證據供畫面讀取", "唯讀後端 · 不修改訓練、決策或 Unity"],
  observer_vue: ["把專案邏輯與活動畫出來", "觀測畫面 · 只顯示，不直接控制正式流程"],

  decision_system: ["根據正式證據決定下一步", "決策與學習 · 不直接定義生產層真相"],
  formal_decision_runtime: ["執行正式決策閉環", "正式流程 · 收件、驗證、候選與唯一裁決"],
  contract_intake: ["接收並驗證生產層事實", "正式收件 · 只接受符合契約的階段事件"],
  candidate_pool: ["整理可執行的下一步候選", "候選管理 · 保留來源、排名與歷史訊號"],
  arbiter: ["選出唯一正式下一步", "仲裁節點 · 避免多個建議同時控制主線"],
  shared_decision: ["把裁決寫回正式決策信箱", "跨層輸出 · 生產層只讀這個正式結果"],
  map_strategy_pack: ["診斷地圖卡在哪一層", "策略包 · 分辨資料、幾何、訓練或部署問題"],
  pointcloud_validator: ["判斷幾何能否進入訓練", "前置閘門 · 阻擋不可靠點雲浪費算力"],
  map_validator: ["判斷訓練結果是否達標", "品質診斷 · 結合固定門檻與歷史趨勢"],
  production_param_gate: ["產生下一輪可執行建議", "參數閘門 · 只輸出建議，不直接覆寫主線"],
  feedback_audit: ["記錄決策最後有沒有用", "回饋閉環 · 追蹤浪費、覆寫與重複問題"],
  current_state: ["整理目前正式狀態", "狀態證據 · 告訴系統現在卡在哪裡"],
  outcome_feedback: ["保存決策後的真實結果", "學習證據 · 讓後續評估知道建議是否有效"],
  learning_curve: ["觀察決策是否逐步變準", "成效趨勢 · 監控成功率、覆寫率與重複錯誤"],
  offline_learning: ["離線吸收歷史經驗", "旁路學習 · 不接管正式 runtime"],
  ollama_teacher: ["替歷史實驗補上語意標籤", "本機教師 · 提供標註與先驗，不直接決策"],
  historical_backfill: ["把舊實驗整理成可學資料", "資料整理 · 補齊缺失標籤與背景脈絡"],
  pytorch_learner: ["從歷史回饋訓練離線模型", "離線學習器 · 只輸出報告供 meta review"],

  governance_system: ["守住正式主線與研究邊界", "治理層 · 防止單次結果或 sandbox 誤升格"],
  formal_docs: ["定義正式規則與共同語言", "治理來源 · 所有執行與判讀必須先遵守"],
  experiment_history: ["保存實驗證據與止損紀錄", "研究記憶 · 避免重跑已證明無效的方向"],
  gate_policy: ["統一各階段的通過規則", "門檻政策 · 避免不同模組各自解讀品質"],

  resources_system: ["提供兩個倉庫共同依賴資源", "執行基礎 · 資料、運算、工具與部署環境"],
  video_frames: ["提供原始影片與影格", "輸入資源 · 所有幾何與品質的最上游證據"],
  colmap_resource: ["提供相機與稀疏幾何工具", "幾何資源 · 支援配對、位姿與場景骨架"],
  gpu_cuda: ["提供訓練與渲染算力", "運算資源 · 支撐長時間高斯訓練"],
  unity_bendviewer: ["提供最終部署檢視環境", "部署資源 · 驗證使用者實際看到的結果"],
  ollama_resource: ["提供本機語意標註能力", "教師資源 · 僅服務離線標註與分析"],
};

createApp({
  data() {
    return {
      snapshot: null,
      liveMetaActivity: null,
      catalog: null,
      error: null,
      lastLoadedAt: null,
      lastMetaEventId: null,
      topologySignature: null,
      pulseNodeId: null,
      selectedNodeId: "reference",
      selectedCatalogNodeId: "mission",
      selectedExperimentId: "formal_mcmc",
      activeView: "architecture",
      architectureMode: "network",
      expandedCatalogNodeIds: [],
      architectureNetwork: {
        nodes: [],
        edges: [],
        particles: [],
        hoverId: null,
        probeUntil: 0,
        probeNodeId: null,
        layoutSignature: "",
      },
      architectureFrame: null,
      timer: null,
      metaTimer: null,
      navItems: [
        { id: "architecture", icon: "樹", label: "Architecture", note: "17 主區塊 / 25 子區塊" },
        { id: "project", icon: "⌘", label: "System Map", note: "單一動態全景圖" },
        { id: "contracts", icon: "⇄", label: "Contract Map", note: "兩個倉庫如何合作" },
        { id: "experiments", icon: "⌁", label: "Experiment Tree", note: "正式 / sandbox / 歷史" },
        { id: "overview", icon: "▶", label: "Runtime", note: "當前工作狀態" },
        { id: "stages", icon: "Ⅱ", label: "Stages", note: "正式執行階段" },
        { id: "activity", icon: "↗", label: "Activity", note: "誰傳給誰" },
        { id: "artifacts", icon: "◇", label: "Artifacts", note: "唯讀正式證據" },
      ],
      stages: [
        { key: "sfm", label: "1A", name: "SfM Geometry" },
        { key: "train", label: "1B", name: "MCMC Training" },
        { key: "export", label: "2", name: "Unity Export" },
      ],
      signalNodes: [
        { id: "video", code: "V", label: "Video / Frames", note: "4K PNG / 原始來源", x: 105, y: 125, tone: "blue" },
        { id: "l0", code: "L0", label: "L0 洗幀與來源 Gate", note: "選幀 / 品質 / 保真", x: 300, y: 125, tone: "blue" },
        { id: "sfm", code: "1A", label: "COLMAP SfM", note: "特徵 / 位姿 / 稀疏幾何", x: 500, y: 125, tone: "blue" },
        { id: "train", code: "1B", label: "MCMC 3DGS", note: "正式訓練 / 離線指標", x: 700, y: 125, tone: "blue" },
        { id: "export", code: "2", label: "Unity PLY Export", note: "座標 / SH / PLY", x: 900, y: 125, tone: "blue" },
        { id: "unity", code: "U", label: "Unity Review", note: "匯入 / 多視角部署檢驗", x: 1100, y: 125, tone: "blue" },
        { id: "reference", code: "R", label: "Reference Gate", note: "同視角品質 / 人工 acceptance", x: 1300, y: 125, tone: "amber" },

        { id: "events", code: "E", label: "Stage Events", note: "latest_*_complete.json", x: 170, y: 385, tone: "amber" },
        { id: "artifact", code: "A", label: "正式 Artifact Bus", note: "event / report / evidence", x: 380, y: 385, tone: "amber" },
        { id: "contract", code: "C", label: "Contract Intake", note: "schema / normalize / validate", x: 590, y: 385, tone: "violet" },
        { id: "validators", code: "V", label: "Validators", note: "問題層診斷", x: 800, y: 385, tone: "violet" },
        { id: "candidate", code: "CP", label: "Candidate Pool", note: "候選排序 / 歷史訊號", x: 1010, y: 385, tone: "violet" },
        { id: "arbiter", code: "D", label: "Agent / Arbiter", note: "唯一正式裁決", x: 1215, y: 385, tone: "violet" },
        { id: "decision_outbox", code: "O", label: "Decision Outbox", note: "latest_*_decision.json", x: 1215, y: 480, tone: "violet" },

        { id: "governance", code: "G", label: "Governance Gate", note: "正式來源 / 升格 / sandbox 邊界", x: 950, y: 570, tone: "amber" },
        { id: "scaffold", code: "S", label: "Scaffold-GS Sandbox", note: "bridge-first / 不接管主線", x: 1215, y: 570, tone: "teal" },
        { id: "dialogue", code: "AI", label: "對話框 AI", note: "meta evaluator / 總分析師", x: 150, y: 680, tone: "cyan" },
        { id: "observer", code: "UI", label: "Vue Observer", note: "唯讀顯示 / 活動路徑", x: 410, y: 680, tone: "green" },
        { id: "human", code: "H", label: "Human Feedback", note: "正式人工回饋 artifact", x: 680, y: 680, tone: "green" },
        { id: "teacher", code: "T", label: "Ollama Teacher", note: "離線語意標註", x: 950, y: 680, tone: "teal" },
        { id: "learner", code: "L", label: "PyTorch Learner", note: "離線模型與報告", x: 1215, y: 680, tone: "teal" },
      ],
      signalEdges: [
        { id: "video__l0", from: "video", to: "l0", path: "M155 125 L250 125", kind: "data" },
        { id: "l0__sfm", from: "l0", to: "sfm", path: "M350 125 L450 125", kind: "data" },
        { id: "sfm__train", from: "sfm", to: "train", path: "M550 125 L650 125", kind: "data" },
        { id: "train__export", from: "train", to: "export", path: "M750 125 L850 125", kind: "data" },
        { id: "export__unity", from: "export", to: "unity", path: "M950 125 L1050 125", kind: "data" },
        { id: "unity__reference", from: "unity", to: "reference", path: "M1150 125 L1250 125", kind: "data" },

        { id: "sfm__events", from: "sfm", to: "events", path: "M500 165 C490 255 300 295 180 345", kind: "formal" },
        { id: "train__events", from: "train", to: "events", path: "M700 165 C630 270 350 300 190 345", kind: "formal" },
        { id: "export__events", from: "export", to: "events", path: "M900 165 C780 290 420 315 200 345", kind: "formal" },
        { id: "events__artifact", from: "events", to: "artifact", path: "M220 385 L330 385", kind: "formal" },
        { id: "artifact__contract", from: "artifact", to: "contract", path: "M430 385 L540 385", kind: "formal" },
        { id: "contract__validators", from: "contract", to: "validators", path: "M640 385 L750 385", kind: "formal" },
        { id: "validators__candidate", from: "validators", to: "candidate", path: "M850 385 L960 385", kind: "formal" },
        { id: "candidate__arbiter", from: "candidate", to: "arbiter", path: "M1060 385 L1165 385", kind: "formal" },
        { id: "arbiter__decision_outbox", from: "arbiter", to: "decision_outbox", path: "M1215 425 L1215 440", kind: "decision" },
        { id: "decision_outbox__l0", from: "decision_outbox", to: "l0", path: "M1165 480 C920 540 430 285 300 165", kind: "decision" },

        { id: "artifact__observer", from: "artifact", to: "observer", path: "M380 425 C380 500 395 585 410 640", kind: "observer" },
        { id: "artifact__dialogue", from: "artifact", to: "dialogue", path: "M350 420 C300 500 210 585 160 640", kind: "observer" },
        { id: "governance__artifact", from: "governance", to: "artifact", path: "M900 570 C700 570 520 500 410 425", kind: "observer" },
        { id: "scaffold__artifact", from: "scaffold", to: "artifact", path: "M1165 570 C850 620 570 530 410 425", kind: "offline" },
        { id: "dialogue__observer", from: "dialogue", to: "observer", path: "M200 680 L360 680", kind: "observer" },
        { id: "reference__human", from: "reference", to: "human", path: "M1300 165 C1250 500 860 570 700 640", kind: "observer" },
        { id: "human__artifact", from: "human", to: "artifact", path: "M650 640 C570 560 460 490 400 425", kind: "observer" },
        { id: "artifact__teacher", from: "artifact", to: "teacher", path: "M410 420 C540 545 780 610 930 640", kind: "offline" },
        { id: "teacher__learner", from: "teacher", to: "learner", path: "M1000 680 L1165 680", kind: "offline" },
        { id: "learner__dialogue", from: "learner", to: "dialogue", path: "M1165 700 C900 760 420 760 200 700", kind: "offline" },
      ],
    };
  },
  computed: {
    activeViewLabel() {
      const labels = {
        architecture: { eyebrow: "治理架構", title: "完整專案治理邏輯" },
        project: { eyebrow: "Live system landscape", title: "雙倉專案動態全景圖" },
        contracts: { eyebrow: "Cross-repository interfaces", title: "兩個倉庫如何共同運作" },
        experiments: { eyebrow: "Research portfolio", title: "實驗方向如何演進" },
        overview: { eyebrow: "Mission topology", title: "專案目前在做什麼" },
        stages: { eyebrow: "Formal runtime", title: "正式階段執行狀態" },
        activity: { eyebrow: "Observer traffic", title: "活動與資料傳遞" },
        artifacts: { eyebrow: "Evidence registry", title: "正式證據監聽" },
      };
      return labels[this.activeView];
    },
    artifacts() {
      return this.snapshot?.artifacts || [];
    },
    catalogBranches() {
      return this.catalog?.architecture || [];
    },
    catalogStats() {
      const systems = this.catalogBranches.length;
      const major = this.catalogBranches.reduce((total, system) => total + (system.children || []).length, 0);
      const children = this.catalogBranches.reduce(
        (total, system) => total + (system.children || []).reduce((subtotal, node) => subtotal + (node.children || []).length, 0),
        0,
      );
      return { systems, major, children, total: systems + major + children };
    },
    architectureFocusTarget() {
      if (!this.catalog) return {};
      const directId = this.latestMetaActivity.target_node;
      if (directId) {
        const direct = this.catalogNodeById(directId);
        if (direct) return direct;
      }
      const evidence = [
        this.latestMetaActivity.to_actor,
        this.latestMetaActivity.title,
        this.latestMetaActivity.summary,
        ...(this.latestMetaActivity.related_artifacts || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      const routes = [
        ["reference_gate", ["reference", "validation_report", "參考影像"]],
        ["deployment_review", ["deployment_review", "部署檢驗"]],
        ["unity_pipeline", ["unity", "ply", "export_ply"]],
        ["training_pipeline", ["train", "3dgs", "mcmc", "lpips", "gsplat"]],
        ["sfm_pipeline", ["sfm", "colmap", "sparse", "matching"]],
        ["l0_pipeline", ["frame", "image source", "preprocess", "洗幀", "png"]],
        ["pytorch_learner", ["pytorch", "learner"]],
        ["ollama_teacher", ["teacher", "ollama", "qwen"]],
        ["formal_decision_runtime", ["arbiter", "decision", "contract_io", "coordinator", "agent_test"]],
        ["map_strategy_pack", ["validator", "strategy", "candidate", "gate"]],
        ["formal_docs", [".md", "docs/", "governance", "文件", "說明書"]],
        ["observer_vue", ["observer", "vue", "app.js", "index.html", "styles.css"]],
        ["feedback_audit", ["feedback", "audit", "review", "meta"]],
      ];
      const match = routes.find(([, terms]) => terms.some((term) => evidence.includes(term)));
      return this.catalogNodeById(match?.[0]) || {};
    },
    architectureFocusPath() {
      return this.catalogPathToNode(this.architectureFocusTarget.id);
    },
    architectureFocusSystem() {
      return this.architectureFocusPath[0] || {};
    },
    architectureFocusMajor() {
      return this.architectureFocusPath[1] || this.architectureFocusSystem;
    },
    architectureFocusTechnicalPath() {
      return this.architectureFocusTarget.path || this.latestMetaActivity.related_artifacts?.[0] || "未提供技術路徑";
    },
    architectureAutoMode() {
      return this.architectureDrillSnapshot().active;
    },
    selectedCatalogNode() {
      if (!this.catalog) return {};
      if (this.selectedCatalogNodeId === this.catalog.mission?.id) return this.catalog.mission;
      for (const branch of this.catalogBranches) {
        const found = this.findCatalogNode(branch, this.selectedCatalogNodeId);
        if (found) return found;
      }
      return this.catalog.mission || {};
    },
    selectedCatalogRepository() {
      const repositoryId = this.selectedCatalogNode.repository || this.findOwningRepository(this.selectedCatalogNodeId);
      return this.repoFor(repositoryId);
    },
    experimentGroups() {
      const definitions = [
        { id: "formal", label: "Formal", title: "正式主線" },
        { id: "planned", label: "Planned", title: "下一步候選" },
        { id: "sandbox", label: "Sandbox", title: "研究支線" },
        { id: "offline", label: "Offline", title: "離線學習" },
        { id: "archived", label: "Archived", title: "已歸檔方向" },
      ];
      return definitions.map((group) => ({
        ...group,
        items: (this.catalog?.experiments || []).filter((item) => item.group === group.id),
      }));
    },
    selectedExperiment() {
      return (this.catalog?.experiments || []).find((item) => item.id === this.selectedExperimentId) || {};
    },
    learner() {
      return this.snapshot?.learner?.scaffold || this.snapshot?.learner?.baseline || {};
    },
    isStale() {
      if (!this.lastLoadedAt) return true;
      return Date.now() - this.lastLoadedAt > 8000;
    },
    healthLabel() {
      if (this.error) return "觀測服務錯誤";
      return this.isStale ? "觀測心跳已過期" : "觀測服務運作中";
    },
    metaActivity() {
      return this.liveMetaActivity || this.snapshot?.meta_activity || {};
    },
    latestMetaActivity() {
      return this.metaActivity.latest || {};
    },
    metaEvents() {
      return this.metaActivity.events || [];
    },
    metaStatusClass() {
      return `status-${this.latestMetaActivity.status || "idle"}`;
    },
    flowFrom() {
      return this.latestMetaActivity.from_actor || this.latestMetaActivity.actor || "dialogue_ai";
    },
    flowTo() {
      return this.latestMetaActivity.to_actor || "observer_ui";
    },
    flowChannel() {
      return this.latestMetaActivity.channel || this.latestMetaActivity.scope || "observer_event";
    },
    signalRouteIsLive() {
      if (!this.latestMetaActivity.event_id) return false;
      if (this.latestMetaActivity.status === "running") return true;
      const timestamp = new Date(this.latestMetaActivity.timestamp || "").getTime();
      return Number.isFinite(timestamp) && Date.now() - timestamp < 60000;
    },
    activeSignalRoute() {
      if (!this.signalRouteIsLive) return null;
      const from = this.signalNodeForActor(this.flowFrom);
      const to = this.signalNodeForActor(this.flowTo);
      if (!from || !to || from.id === to.id) return null;
      return {
        from,
        to,
        edgeId: `${from.id}__${to.id}`,
        channel: this.flowChannel,
        title: this.latestMetaActivity.title || "Latest observer traffic",
      };
    },
    renderedSignalEdges() {
      const edges = [...this.signalEdges];
      const route = this.activeSignalRoute;
      if (route && !edges.some((edge) => edge.id === route.edgeId)) {
        edges.push({
          id: route.edgeId,
          from: route.from.id,
          to: route.to.id,
          path: this.directSignalPath(route.from, route.to),
          kind: "dynamic",
        });
      }
      return edges;
    },
    bestLpips() {
      const value = this.snapshot?.mcmc?.best_lpips_run?.lpips;
      return typeof value === "number" ? value.toFixed(6) : "-";
    },
    bestRun() {
      return this.snapshot?.mcmc?.best_lpips_run?.run_name || "formal benchmark";
    },
    currentTrainLpips() {
      const value = this.eventFor("train").metrics?.lpips;
      return typeof value === "number" ? value.toFixed(6) : "-";
    },
    productionNodes() {
      const sfm = this.eventFor("sfm");
      const review = this.snapshot?.deployment_review || {};
      const reviewPass = review.deployment_review_pass === true || review.verdict?.deployment_review_pass === true;
      const deployablePass = review.deployable_pass === true || review.verdict?.deployable_pass === true;
      return [
        {
          id: "source",
          step: "L0",
          kicker: "影像來源",
          title: "4K PNG Frames",
          subtitle: sfm.run_id ? "正式 SfM 已消費影像來源" : "等待正式來源",
          status: sfm.run_id ? "ready" : "missing",
          stateLabel: sfm.run_id ? "已進鏈" : "未執行",
          detail: "保留影像來源並送入 SfM。來源品質是正式變因，但不能直接代表 Unity 品質。",
          evidence: sfm.run_root || "latest_sfm_complete.json missing",
          boundary: "生產層輸入；不因單次人工觀察改寫來源規則。",
        },
        this.stageNode("sfm", "1A", "SfM Geometry", "COLMAP / 相機位姿與稀疏幾何"),
        this.stageNode("train", "1B", "MCMC Training", "3DGS 訓練與離線品質指標"),
        this.stageNode("export", "2", "Unity Export", "PLY / 座標系 / SH 轉換"),
        {
          id: "unity",
          step: "U",
          kicker: "部署技術檢驗",
          title: "Unity Review",
          subtitle: reviewPass ? "技術 Gate 通過" : "等待通過部署檢驗",
          status: deployablePass ? "ready" : reviewPass ? "review" : Object.keys(review).length ? "blocked" : "missing",
          stateLabel: deployablePass ? "可交付" : reviewPass ? "技術通過" : Object.keys(review).length ? "阻塞" : "未執行",
          detail: reviewPass
            ? "確認 Unity 畫面沒有黑屏、嚴重霧化或匯入失敗；尚不能證明與真實場景一致。"
            : "尚未取得通過的 Unity deployment review。",
          evidence: review.output_dir || review.run_root || "latest deployment_review.json missing",
          boundary: "生產層部署檢查；技術通過不能直接升格 deployable。",
        },
        this.referenceNode,
      ];
    },
    referenceNode() {
      const review = this.snapshot?.deployment_review || {};
      const deployablePass = review.deployable_pass === true || review.verdict?.deployable_pass === true;
      return {
        id: "reference",
        step: "R",
        kicker: "最終品質判定",
        title: "Reference Gate",
        subtitle: deployablePass ? "同視角 reference 證據完成" : "缺少同視角真實照片比對",
        status: deployablePass ? "ready" : "blocked",
        stateLabel: deployablePass ? "可交付" : "核心阻塞",
        detail: deployablePass
          ? "正式 evidence 已允許 deployable promotion。"
          : "必須用相同相機視角比較 Reference、原生 3DGS render 與 Unity render，才能定位渲染品質損失。",
        evidence: deployablePass ? "deployment_review deployable_pass=true" : "unity_reference_validation_report.json pending",
        boundary: "升格 Gate；沒有 reference 或正式人工 acceptance，不得宣稱可交付。",
      };
    },
    evidenceDecisionNodes() {
      const exportDecision = this.decisionFor("export");
      const hasArtifacts = this.artifacts.some((item) => item.exists);
      const hasDecision = Boolean(exportDecision.decision);
      return [
        {
          id: "artifact",
          step: "A",
          kicker: "正式證據",
          title: "Artifact Bus",
          subtitle: "events / reports / deployment review",
          status: hasArtifacts ? "ready" : "missing",
          stateLabel: hasArtifacts ? "持續監聽" : "缺少證據",
          detail: "保存生產層產出的可追蹤事實，避免用聊天內容或舊路徑覆蓋主線。",
          evidence: `${this.artifacts.filter((item) => item.exists).length} watched artifacts available`,
          boundary: "正式證據層；只保存與傳遞事實，不裁決下一步。",
        },
        {
          id: "validators",
          step: "V",
          kicker: "問題診斷",
          title: "Validators",
          subtitle: "讀 contract / report，提出候選",
          status: hasDecision ? "ready" : "missing",
          stateLabel: hasDecision ? "已分析" : "未執行",
          detail: "根據正式 evidence 判斷問題層並提出候選，不直接改生產主線。",
          evidence: exportDecision.reason || "latest_export_decision.json missing",
          boundary: "決策層候選產生器；沒有正式控制權。",
        },
        {
          id: "arbiter",
          step: "D",
          kicker: "唯一正式裁決",
          title: "Agent / Arbiter",
          subtitle: exportDecision.decision || "等待 decision",
          status: hasDecision ? (exportDecision.requires_human_review ? "review" : "ready") : "missing",
          stateLabel: hasDecision ? (exportDecision.requires_human_review ? "需人工審查" : "已有裁決") : "未執行",
          detail: "從候選池選出唯一正式下一步，再透過 latest_*_decision 回寫生產層。",
          evidence: exportDecision.reason || "latest_export_decision.json missing",
          boundary: "唯一可輸出正式 next step 的決策元件。",
        },
      ];
    },
    sidecarNodes() {
      const teacherStatus = this.snapshot?.teacher?.status;
      const hasLearner = Object.keys(this.learner).length > 0;
      return [
        {
          id: "observer",
          step: "UI",
          kicker: "唯讀觀測",
          title: "Vue Observer",
          subtitle: "讀取正式 artifact",
          status: this.isStale ? "blocked" : "ready",
          stateLabel: this.isStale ? "連線異常" : "觀測中",
          detail: "顯示專案狀態與資料流，不控制 training、Unity 或正式決策。",
          evidence: this.snapshot?.observer?.heartbeat || "no heartbeat",
          boundary: "只讀 sidecar；禁止寫入 latest_*_decision.json。",
        },
        {
          id: "dialogue",
          step: "AI",
          kicker: "總分析師",
          title: "Dialogue AI",
          subtitle: "Meta evaluator",
          status: this.latestMetaActivity.event_id ? "ready" : "missing",
          stateLabel: this.latestMetaActivity.event_id ? "有活動" : "等待活動",
          detail: "統整正式證據並審查實驗、teacher、learner 與升格條件。",
          evidence: this.latestMetaActivity.title || "no meta activity",
          boundary: "單次對話不能取代正式 artifact 或決策。",
        },
        {
          id: "offline",
          step: "ML",
          kicker: "離線學習",
          title: "Teacher / Learner",
          subtitle: "Ollama labels / PyTorch reports",
          status: teacherStatus && hasLearner ? "ready" : teacherStatus || hasLearner ? "review" : "missing",
          stateLabel: teacherStatus && hasLearner ? "報告可用" : "未完整執行",
          detail: "Teacher 提供語意標註，Learner 吸收歷史 feedback；兩者都不能覆寫正式 decision。",
          evidence: hasLearner ? `dataset_size=${this.learner.dataset_size ?? "unknown"}` : "offline learner report missing",
          boundary: "離線 sidecar；只輸出標註與報告。",
        },
        {
          id: "human",
          step: "H",
          kicker: "人工閉環",
          title: "Human Feedback",
          subtitle: "acceptance / deployment note",
          status: "review",
          stateLabel: "待正式落檔",
          detail: "人工觀察只有落成 deployment review 或 outcome feedback，才可進入學習與升格審查。",
          evidence: "formal feedback artifact required",
          boundary: "人工輸入必須先落成正式 artifact。",
        },
      ];
    },
    allNodes() {
      return [...this.productionNodes, ...this.evidenceDecisionNodes, ...this.sidecarNodes];
    },
    selectedNode() {
      return this.allNodes.find((node) => node.id === this.selectedNodeId) || this.referenceNode;
    },
  },
  mounted() {
    this.loadCatalog();
    this.loadSnapshot();
    this.loadMetaActivity();
    this.timer = window.setInterval(this.loadSnapshot, 2500);
    this.metaTimer = window.setInterval(this.loadMetaActivity, 600);
    window.addEventListener("resize", this.resizeArchitectureNetwork);
  },
  unmounted() {
    if (this.timer) window.clearInterval(this.timer);
    if (this.metaTimer) window.clearInterval(this.metaTimer);
    if (this.architectureFrame) window.cancelAnimationFrame(this.architectureFrame);
    window.removeEventListener("resize", this.resizeArchitectureNetwork);
  },
  methods: {
    async loadCatalog() {
      try {
        const response = await fetch("/api/catalog", { cache: "no-store" });
        if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
        this.catalog = await response.json();
        if (!this.expandedCatalogNodeIds.length) this.expandAllCatalogNodes();
        await this.$nextTick();
        this.initArchitectureNetwork();
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      }
    },
    async loadSnapshot() {
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const nextSnapshot = await response.json();
        const nextMetaId = nextSnapshot?.meta_activity?.latest?.event_id || null;

        this.snapshot = nextSnapshot;
        this.error = null;
        this.lastLoadedAt = Date.now();

        const nextSignature = JSON.stringify(this.allNodes.map((node) => [node.id, node.status, node.stateLabel]));
        if (this.topologySignature && this.topologySignature !== nextSignature) {
          const previous = JSON.parse(this.topologySignature);
          const current = JSON.parse(nextSignature);
          const changed = current.find((entry, index) => JSON.stringify(entry) !== JSON.stringify(previous[index]));
          this.pulseNodeId = changed?.[0] || null;
          window.setTimeout(() => {
            this.pulseNodeId = null;
          }, 1300);
        } else if (nextMetaId && nextMetaId !== this.lastMetaEventId) {
          this.pulseNodeId = "dialogue";
          window.setTimeout(() => {
            this.pulseNodeId = null;
          }, 1300);
        }
        this.topologySignature = nextSignature;
        this.lastMetaEventId = nextMetaId;
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      }
    },
    async loadMetaActivity() {
      try {
        const response = await fetch("/api/meta-activity", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        this.liveMetaActivity = payload?.meta_activity || null;
      } catch {
        // Full snapshot health remains the source for observer connection errors.
      }
    },
    eventFor(stage) {
      return this.snapshot?.formal_runtime?.events?.[stage] || {};
    },
    decisionFor(stage) {
      return this.snapshot?.formal_runtime?.decisions?.[stage] || {};
    },
    repoFor(repositoryId) {
      if (!repositoryId) return null;
      return (this.catalog?.repositories || []).find((repo) => repo.id === repositoryId) || null;
    },
    findCatalogNode(node, id) {
      if (!node) return null;
      if (node.id === id) return node;
      for (const child of node.children || []) {
        const found = this.findCatalogNode(child, id);
        if (found) return found;
      }
      return null;
    },
    findOwningRepository(id) {
      for (const branch of this.catalogBranches) {
        if (this.findCatalogNode(branch, id)) return branch.repository || null;
      }
      return null;
    },
    catalogNodeCount(node) {
      return (node?.children || []).length;
    },
    catalogNodeCode(node) {
      if (ARCHITECTURE_ABBREVIATIONS[node?.id]) return ARCHITECTURE_ABBREVIATIONS[node.id];
      const preferred = node?.name_zh || node?.name || node?.id || "?";
      const ascii = String(node?.name || "").match(/[A-Za-z0-9]/g);
      if (ascii?.length) return ascii.slice(0, 2).join("").toUpperCase();
      return String(preferred).slice(0, 1);
    },
    catalogNodeById(id) {
      if (!id) return null;
      for (const system of this.catalogBranches) {
        if (system.id === id) return system;
        for (const major of system.children || []) {
          if (major.id === id) return major;
          const child = (major.children || []).find((item) => item.id === id);
          if (child) return child;
        }
      }
      return null;
    },
    catalogPathToNode(id) {
      if (!id) return [];
      for (const system of this.catalogBranches) {
        if (system.id === id) return [system];
        for (const major of system.children || []) {
          if (major.id === id) return [system, major];
          const child = (major.children || []).find((item) => item.id === id);
          if (child) return [system, major, child];
        }
      }
      return [];
    },
    architectureDrillSnapshot() {
      const timestamp = new Date(this.latestMetaActivity.timestamp || "").getTime();
      const age = Number.isFinite(timestamp) ? Math.max(0, Date.now() - timestamp) : Number.POSITIVE_INFINITY;
      const status = this.latestMetaActivity.status || "idle";
      const path = this.catalogPathToNode(this.architectureFocusTarget.id);
      if (!path.length) return { active: false, depth: 0, path, status, age };
      if (status === "running" && age < 90000) {
        const depth = age < 1400 ? 0 : age < 2800 ? 1 : 2;
        return { active: true, depth: Math.min(depth, path.length - 1), path, status, age };
      }
      if (["warning", "failed"].includes(status) && age < 20000) {
        return { active: true, depth: Math.min(2, path.length - 1), path, status, age };
      }
      if (age < 5000) return { active: true, depth: Math.min(2, path.length - 1), path, status, age };
      if (age < 7500) return { active: true, depth: Math.min(1, path.length - 1), path, status, age };
      return { active: false, depth: 0, path, status, age };
    },
    architectureLogicTitle(node) {
      return ARCHITECTURE_LOGIC_LABELS[node?.id]?.[0] || node?.name_zh || node?.name || node?.id || "未命名節點";
    },
    architectureLogicDetail(node) {
      const typeLabels = {
        system: "系統層",
        component: "功能區塊",
        capability: "執行能力",
        artifact: "正式證據",
        gate: "品質閘門",
        resource: "執行資源",
      };
      return ARCHITECTURE_LOGIC_LABELS[node?.id]?.[1] || `${typeLabels[node?.type] || "治理節點"} · 顯示正式責任與資料作用`;
    },
    actorDisplayName(actor) {
      const labels = {
        dialogue_ai: "對話框 AI",
        codex_meta_evaluator: "對話框 AI",
        observer_ui: "Vue 觀測層",
        production_repo: "生產層",
        decision_repo: "決策層",
        human: "人工回饋",
        ollama_teacher: "Ollama 教師",
        pytorch_learner: "PyTorch 學習器",
      };
      return labels[String(actor || "").toLowerCase()] || actor || "未知節點";
    },
    channelDisplayName(channel) {
      const labels = {
        observer_event: "觀測活動事件",
        formal_event: "正式階段事件",
        formal_decision: "正式決策",
        outcome_feedback: "結果回饋",
      };
      return labels[String(channel || "").toLowerCase()] || channel || "未知通道";
    },
    isCatalogExpanded(id) {
      return this.expandedCatalogNodeIds.includes(id);
    },
    toggleCatalogNode(id) {
      this.expandedCatalogNodeIds = this.isCatalogExpanded(id)
        ? this.expandedCatalogNodeIds.filter((item) => item !== id)
        : [...this.expandedCatalogNodeIds, id];
    },
    expandAllCatalogNodes() {
      this.expandedCatalogNodeIds = this.catalogBranches.flatMap((system) => [
        system.id,
        ...(system.children || []).map((node) => node.id),
      ]);
    },
    collapseAllCatalogNodes() {
      this.expandedCatalogNodeIds = [];
    },
    setArchitectureMode(mode) {
      this.architectureMode = mode;
      if (mode === "network") {
        this.$nextTick(() => this.initArchitectureNetwork());
      }
    },
    initArchitectureNetwork() {
      const canvas = this.$refs.architectureCanvas;
      if (!canvas || !this.catalog) return;
      this.resizeArchitectureNetwork();
      if (!this.architectureFrame) this.drawArchitectureNetwork();
    },
    resizeArchitectureNetwork() {
      const canvas = this.$refs.architectureCanvas;
      if (!canvas || this.architectureMode !== "network") return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(900, Math.round(rect.width * dpr));
      canvas.height = Math.max(640, Math.round(rect.height * dpr));
      this.buildArchitectureNetwork(canvas.width / dpr, canvas.height / dpr);
    },
    buildArchitectureNetwork(width, height) {
      if (!this.catalog) return;
      const nodes = [];
      const edges = [];
      const drill = this.architectureDrillSnapshot();
      const addNode = (node, level, x, y, w, h, parentId = null, clusterId = node.id) => {
        nodes.push({ ...node, level, x, y, w, h, r: level === "system" ? 22 : level === "major" ? 16 : 12, parentId, clusterId });
      };
      const addEdge = (from, to, kind = "hierarchy") => edges.push({ id: `${from}__${to}`, from, to, kind });

      if (drill.depth === 0) {
        const cardWidth = Math.min(420, (width - 150) / 2);
        const cardHeight = 150;
        const positions = [
          [width * 0.27, height * 0.3],
          [width * 0.73, height * 0.3],
          [width * 0.27, height * 0.7],
          [width * 0.73, height * 0.7],
        ];
        this.catalogBranches.forEach((system, index) => addNode(system, "system", positions[index][0], positions[index][1], cardWidth, cardHeight));
      } else if (drill.depth === 1) {
        const system = drill.path[0];
        addNode(system, "system", width / 2, 92, Math.min(640, width * 0.66), 118);
        const majors = system.children || [];
        const columns = majors.length > 4 ? 3 : 2;
        const rows = Math.ceil(majors.length / columns);
        const gapX = 24;
        const gapY = 28;
        const majorWidth = Math.min(330, (width - 90 - gapX * (columns - 1)) / columns);
        const majorHeight = 105;
        majors.forEach((major, index) => {
          const column = index % columns;
          const row = Math.floor(index / columns);
          const totalWidth = columns * majorWidth + (columns - 1) * gapX;
          const x = (width - totalWidth) / 2 + majorWidth / 2 + column * (majorWidth + gapX);
          const y = 235 + row * (majorHeight + gapY);
          addNode(major, "major", x, y, majorWidth, majorHeight, system.id, system.id);
          addEdge(system.id, major.id);
        });
      } else {
        const [system, major, target] = drill.path;
        addNode(system, "system", width / 2, 72, Math.min(600, width * 0.64), 94);
        addNode(major, "major", width / 2, 210, Math.min(520, width * 0.56), 104, system.id, system.id);
        addEdge(system.id, major.id);
        const children = major.children || [];
        const columns = children.length > 3 ? 2 : Math.max(children.length, 1);
        const childWidth = Math.min(340, (width - 100 - 24 * (columns - 1)) / columns);
        const childHeight = 94;
        children.forEach((child, index) => {
          const column = index % columns;
          const row = Math.floor(index / columns);
          const totalWidth = columns * childWidth + (columns - 1) * 24;
          const x = (width - totalWidth) / 2 + childWidth / 2 + column * (childWidth + 24);
          const y = 370 + row * 126;
          addNode(child, "child", x, y, childWidth, childHeight, major.id, system.id);
          addEdge(major.id, child.id, "hierarchy-child");
        });
        if (target && target.id === major.id) this.architectureNetwork.probeNodeId = major.id;
      }

      if (this.architectureAutoMode) {
        const source = {
          id: "dialogue_ai_source",
          name: "Dialogue AI",
          name_zh: "對話框 AI",
          type: "sidecar",
          status: "active",
          path: "observer activity",
        };
        addNode(source, "source", 118, height - 80, 190, 72, null, "observer_system");
      }
      this.architectureNetwork.nodes = nodes;
      this.architectureNetwork.edges = edges;
      this.architectureNetwork.layoutSignature = `${drill.depth}:${drill.path.map((node) => node.id).join("/")}:${width}:${height}`;
    },
    architectureNodeAt(event) {
      const canvas = this.$refs.architectureCanvas;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      return [...this.architectureNetwork.nodes]
        .reverse()
        .find((node) => Math.abs(node.x - x) <= node.w / 2 + 5 && Math.abs(node.y - y) <= node.h / 2 + 5);
    },
    architectureNodeById(id) {
      return this.architectureNetwork.nodes.find((node) => node.id === id) || null;
    },
    architecturePointerMove(event) {
      const node = this.architectureNodeAt(event);
      this.architectureNetwork.hoverId = node?.id || null;
      event.currentTarget.style.cursor = node ? "pointer" : "default";
    },
    architecturePointerLeave() {
      this.architectureNetwork.hoverId = null;
    },
    architecturePointerClick(event) {
      const node = this.architectureNodeAt(event);
      if (!node) return;
      this.selectedCatalogNodeId = node.id;
      this.architectureNetwork.probeNodeId = node.id;
      this.architectureNetwork.probeUntil = Date.now() + 1800;
      const connected = this.architectureNetwork.edges.filter((edge) => edge.from === node.id || edge.to === node.id);
      connected.forEach((edge, index) => {
        window.setTimeout(() => this.spawnArchitectureParticles(edge, 4), index * 80);
      });
    },
    architectureNodeForActor(actor) {
      const value = String(actor || "").toLowerCase();
      const aliases = [
        ["observer_system", ["observer", "vue", "dialogue_ai", "dialogue", "codex", "觀測"]],
        ["offline_learning", ["teacher", "learner", "ollama", "qwen", "pytorch"]],
        ["formal_decision_runtime", ["agent", "arbiter", "decision", "決策"]],
        ["unity_pipeline", ["unity", "deployment"]],
        ["sfm_pipeline", ["sfm", "colmap"]],
        ["training_pipeline", ["train", "mcmc", "3dgs"]],
        ["l0_pipeline", ["frame", "l0", "production"]],
        ["feedback_audit", ["human", "feedback", "meta"]],
      ];
      const match = aliases.find(([, terms]) => terms.some((term) => value.includes(term)));
      return this.architectureNetwork.nodes.find((node) => node.id === match?.[0]) || null;
    },
    architectureNodeForActivity(activity) {
      if (activity?.target_node) {
        const direct = this.catalogNodeById(activity.target_node);
        if (direct) return direct;
      }
      const evidence = [
        activity?.to_actor,
        activity?.title,
        activity?.summary,
        ...(activity?.related_artifacts || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      const routes = [
        ["unity_pipeline", ["unity", "deployment", "ply", "export_ply"]],
        ["training_pipeline", ["train", "3dgs", "mcmc", "lpips", "gsplat"]],
        ["sfm_pipeline", ["sfm", "colmap", "sparse", "matching"]],
        ["l0_pipeline", ["frame", "image source", "preprocess", "洗幀", "png"]],
        ["offline_learning", ["teacher", "learner", "ollama", "qwen", "pytorch", "offline_learning"]],
        ["formal_decision_runtime", ["arbiter", "decision", "contract_io", "coordinator", "agent_test"]],
        ["map_strategy_pack", ["validator", "strategy", "candidate", "gate"]],
        ["formal_docs", [".md", "docs/", "governance", "文件", "說明書"]],
        ["observer_system", ["observer", "vue", "app.js", "index.html", "styles.css"]],
        ["feedback_audit", ["feedback", "audit", "review", "meta"]],
      ];
      const match = routes.find(([, terms]) => terms.some((term) => evidence.includes(term)));
      return this.catalogNodeById(match?.[0]) || this.architectureNodeForActor(activity?.to_actor);
    },
    architectureActiveEdge() {
      if (!this.architectureAutoMode) return null;
      const fromNode = this.architectureNetwork.nodes.find((node) => node.id === "dialogue_ai_source");
      const drill = this.architectureDrillSnapshot();
      const visibleTarget = [...drill.path].reverse().find((node) => this.architectureNetwork.nodes.some((visible) => visible.id === node.id));
      const toNode = this.architectureNetwork.nodes.find((node) => node.id === visibleTarget?.id);
      if (!fromNode || !toNode || fromNode.id === toNode.id) return null;
      return { id: "live-route", from: fromNode.id, to: toNode.id, kind: "live" };
    },
    spawnArchitectureParticles(edge, count = 5) {
      const from = this.architectureNetwork.nodes.find((node) => node.id === edge.from);
      const to = this.architectureNetwork.nodes.find((node) => node.id === edge.to);
      if (!from || !to) return;
      for (let index = 0; index < count; index += 1) {
        this.architectureNetwork.particles.push({
          fromX: from.x,
          fromY: from.y,
          toX: to.x,
          toY: to.y,
          kind: edge.kind,
          progress: -index * 0.13,
          speed: 0.012 + Math.random() * 0.004,
        });
      }
    },
    drawArchitectureNetwork() {
      const canvas = this.$refs.architectureCanvas;
      if (!canvas || this.architectureMode !== "network") {
        this.architectureFrame = null;
        return;
      }
      const context = canvas.getContext("2d");
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;
      const drill = this.architectureDrillSnapshot();
      const expectedSignature = `${drill.depth}:${drill.path.map((node) => node.id).join("/")}:${width}:${height}`;
      if (this.architectureNetwork.layoutSignature !== expectedSignature) this.buildArchitectureNetwork(width, height);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, width, height);
      this.drawArchitectureBackground(context, width, height);

      const activeEdge = this.architectureActiveEdge();
      const edges = activeEdge ? [...this.architectureNetwork.edges, activeEdge] : this.architectureNetwork.edges;
      edges.forEach((edge) => this.drawArchitectureEdge(context, edge, activeEdge?.id === edge.id));

      if (activeEdge && !this.architectureNetwork.particles.some((particle) => particle.kind === "live")) {
        this.spawnArchitectureParticles(activeEdge, 7);
      }

      this.architectureNetwork.nodes.forEach((node) => this.drawArchitectureNode(context, node));
      this.drawArchitectureParticles(context);
      this.architectureFrame = window.requestAnimationFrame(() => this.drawArchitectureNetwork());
    },
    drawArchitectureBackground(context, width, height) {
      context.save();
      context.fillStyle = "#07101d";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "rgba(90, 167, 255, 0.08)";
      context.lineWidth = 0.5;
      for (let x = 0; x < width; x += 44) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let y = 0; y < height; y += 44) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }
      const systems = this.architectureNetwork.nodes.filter((node) => node.level === "system");
      systems.forEach((system) => {
          const color = this.architectureNodeColor(system);
          context.save();
          const columnLeft = system.x - system.w / 2 - 9;
          const columnWidth = system.w + 18;
          context.strokeStyle = `${color}36`;
          context.fillStyle = `${color}07`;
          context.setLineDash([5, 9]);
          context.lineWidth = 1;
          this.roundedArchitectureRect(context, columnLeft, 16, columnWidth, height - 32, 16);
          context.fill();
          context.stroke();
          context.restore();
        });
      context.restore();
    },
    roundedArchitectureRect(context, x, y, width, height, radius) {
      const safeRadius = Math.min(radius, width / 2, height / 2);
      context.beginPath();
      context.moveTo(x + safeRadius, y);
      context.arcTo(x + width, y, x + width, y + height, safeRadius);
      context.arcTo(x + width, y + height, x, y + height, safeRadius);
      context.arcTo(x, y + height, x, y, safeRadius);
      context.arcTo(x, y, x + width, y, safeRadius);
      context.closePath();
    },
    architectureText(value, maxLength) {
      const text = String(value || "");
      return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
    },
    architectureNodeColor(node) {
      const clusterColors = {
        production_system: "#5aa7ff",
        decision_system: "#e87b22",
        governance_system: "#42d392",
        resources_system: "#9f8cff",
      };
      return clusterColors[node?.clusterId || node?.id] || this.architectureColor(node?.status);
    },
    architectureColor(kind) {
      const colors = {
        formal: "#42d392",
        active: "#5aa7ff",
        offline: "#9f8cff",
        resource: "#58d6d1",
        planned: "#f2b84b",
        hierarchy: "#35516e",
        "hierarchy-child": "#21364d",
        production: "#388bfd",
        contract: "#e3b341",
        decision: "#8957e5",
        sidecar: "#2ecc71",
        live: "#ffffff",
      };
      return colors[kind] || "#8794a8";
    },
    drawArchitectureEdge(context, edge, active = false) {
      const from = this.architectureNetwork.nodes.find((node) => node.id === edge.from);
      const to = this.architectureNetwork.nodes.find((node) => node.id === edge.to);
      if (!from || !to) return;
      const probeActive =
        Date.now() < this.architectureNetwork.probeUntil &&
        (edge.from === this.architectureNetwork.probeNodeId || edge.to === this.architectureNetwork.probeNodeId);
      const color = edge.kind.startsWith("hierarchy") ? this.architectureNodeColor(from) : this.architectureColor(edge.kind);
      context.save();
      context.globalAlpha = active ? 0.95 : probeActive ? 0.75 : edge.kind.startsWith("hierarchy") ? 0.22 : 0.28;
      context.strokeStyle = color;
      context.lineWidth = active ? 2.8 : probeActive ? 2 : edge.kind === "hierarchy-child" ? 0.8 : edge.kind === "hierarchy" ? 1 : 1.35;
      if (edge.kind === "decision") context.setLineDash([8, 6]);
      if (edge.kind === "sidecar") context.setLineDash([3, 7]);
      if (edge.kind === "resource") context.setLineDash([2, 5]);
      if (active || probeActive) {
        context.shadowColor = color;
        context.shadowBlur = 10;
      }
      if (edge.kind.startsWith("hierarchy")) {
        const fromX = from.x;
        const fromY = from.y + from.h / 2;
        const toX = to.x;
        const toY = to.y - to.h / 2;
        const midY = (fromY + toY) / 2;
        context.beginPath();
        context.moveTo(fromX, fromY);
        context.lineTo(fromX, midY);
        context.lineTo(toX, midY);
        context.lineTo(toX, toY);
        context.stroke();
        context.restore();
        return;
      }
      const fromX = from.x + (to.x >= from.x ? from.w / 2 : -from.w / 2);
      const toX = to.x + (to.x >= from.x ? -to.w / 2 : to.w / 2);
      const midX = (fromX + toX) / 2;
      const midY = (from.y + to.y) / 2 - Math.min(46, Math.abs(toX - fromX) * 0.045);
      context.beginPath();
      context.moveTo(fromX, from.y);
      context.quadraticCurveTo(midX, midY, toX, to.y);
      context.stroke();
      if (!edge.kind.startsWith("hierarchy") && edge.kind !== "resource") {
        const t = 0.92;
        const x = (1 - t) * (1 - t) * fromX + 2 * (1 - t) * t * midX + t * t * toX;
        const y = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * midY + t * t * to.y;
        const dx = 2 * (1 - t) * (midX - fromX) + 2 * t * (toX - midX);
        const dy = 2 * (1 - t) * (midY - from.y) + 2 * t * (to.y - midY);
        const angle = Math.atan2(dy, dx);
        context.translate(x, y);
        context.rotate(angle);
        context.fillStyle = color;
        context.beginPath();
        context.moveTo(0, 0);
        context.lineTo(-7, -3);
        context.lineTo(-7, 3);
        context.closePath();
        context.fill();
      }
      context.restore();
    },
    drawArchitectureNode(context, node) {
      const hovered = this.architectureNetwork.hoverId === node.id;
      const selected = this.selectedCatalogNodeId === node.id;
      const probed = Date.now() < this.architectureNetwork.probeUntil && this.architectureNetwork.probeNodeId === node.id;
      const activeEdge = this.architectureActiveEdge();
      const activitySource = activeEdge?.from === node.id;
      const activityTarget = activeEdge?.to === node.id;
      const focused = hovered || selected || probed || activitySource || activityTarget;
      const clusterColor = this.architectureNodeColor(node);
      const color = activitySource ? "#388bfd" : activityTarget ? "#2ecc71" : clusterColor;
      const statusColor = this.architectureColor(node.status);
      const pulse = activitySource || activityTarget ? 1.06 + Math.sin(Date.now() * 0.006) * 0.025 : 1;
      const focusScale = selected || probed ? 1.08 : hovered ? 1.045 : 1;
      const width = node.w * focusScale * pulse;
      const height = node.h * focusScale * pulse;
      context.save();
      if (focused) {
        context.shadowColor = color;
        context.shadowBlur = 24;
      }
      context.fillStyle = focused ? `${color}33` : "#0d1928";
      context.strokeStyle = color;
      context.globalAlpha = node.level === "child" && !focused ? 0.88 : 1;
      context.lineWidth = node.level === "system" ? 2.2 : focused ? 1.8 : 1;
      this.roundedArchitectureRect(context, node.x - width / 2, node.y - height / 2, width, height, node.r);
      context.fill();
      context.stroke();
      context.fillStyle = statusColor;
      context.globalAlpha = 1;
      this.roundedArchitectureRect(context, node.x - width / 2, node.y - height / 2, width, 4, 2);
      context.fill();
      if (activitySource || activityTarget) {
        context.strokeStyle = "#ffffff";
        context.globalAlpha = 0.78;
        context.lineWidth = 1.4;
        this.roundedArchitectureRect(context, node.x - width / 2 - 5, node.y - height / 2 - 5, width + 10, height + 10, node.r + 4);
        context.stroke();
        context.globalAlpha = 1;
      }
      context.textAlign = "left";
      context.textBaseline = "middle";
      const left = node.x - width / 2 + (node.level === "child" ? 8 : 11);
      const top = node.y - height / 2;
      context.fillStyle = focused ? "#ffffff" : "#e8edf5";
      context.font = `800 ${node.level === "system" ? 12 : node.level === "major" ? 10 : 8}px Cascadia Code`;
      context.fillText(this.catalogNodeCode(node), left, top + height * 0.42);
      const codeWidth = context.measureText(this.catalogNodeCode(node)).width;
      context.fillStyle = focused ? "#ffffff" : "#bdc7d5";
      context.font = `600 ${node.level === "system" ? 13 : node.level === "major" ? 10 : 8}px Microsoft JhengHei`;
      context.fillText(
        this.architectureText(this.architectureLogicTitle(node), node.level === "child" ? 17 : 25),
        left + codeWidth + 8,
        top + height * 0.42,
      );
      context.fillStyle = focused ? "#cbd4e2" : "#8794a8";
      context.font = `${node.level === "system" ? 9 : node.level === "major" ? 8 : 7}px Microsoft JhengHei`;
      context.fillText(
        this.architectureText(this.architectureLogicDetail(node), node.level === "child" ? 25 : 40),
        left,
        top + height * 0.74,
      );
      context.restore();
    },
    drawArchitectureParticles(context) {
      const colors = { live: "#ffffff", production: "#388bfd", contract: "#e3b341", decision: "#8957e5", sidecar: "#2ecc71" };
      this.architectureNetwork.particles = this.architectureNetwork.particles.filter((particle) => particle.progress < 1);
      this.architectureNetwork.particles.forEach((particle) => {
        particle.progress += particle.speed;
        if (particle.progress < 0) return;
        const t = particle.progress;
        const x = particle.fromX + (particle.toX - particle.fromX) * t;
        const y = particle.fromY + (particle.toY - particle.fromY) * t;
        const color = colors[particle.kind] || this.architectureColor(particle.kind);
        context.save();
        context.fillStyle = color;
        context.shadowColor = color;
        context.shadowBlur = 12;
        context.globalAlpha = t > 0.82 ? (1 - t) / 0.18 : 1;
        context.beginPath();
        context.arc(x, y, particle.kind === "live" ? 3 : 2.2, 0, Math.PI * 2);
        context.fill();
        context.restore();
      });
    },
    catalogNodeRole(node) {
      const labels = {
        mission: "整個雙倉專案的共同使命。",
        system: "跨元件的穩定責任邊界。",
        component: "可獨立維護與測試的主要程式模組。",
        capability: "元件提供的具體能力。",
        artifact: "跨階段或跨倉傳遞的正式證據。",
        gate: "決定是否可繼續或升格的品質閥門。",
        resource: "系統依賴的資料、運算或部署資源。",
      };
      return labels[node.type] || "專案架構節點。";
    },
    contractActor(actor) {
      if (actor === "human") return "Human";
      return this.repoFor(actor)?.name || actor;
    },
    experimentGovernance(experiment) {
      const rules = {
        formal: "可作正式 baseline，但仍需遵守 Gate 與 deployment review。",
        planned: "尚未執行或未完成 artifact，不得升格。",
        sandbox: "只能研究與驗證 bridge，不得接管 formal runtime。",
        offline: "只吸收標註與 feedback，不得覆寫正式 decision。",
        archived: "已有負面或收斂結論，除非提出新公平假設不得重跑。",
      };
      return rules[experiment.group] || "需由 meta evaluator 審查後決定。";
    },
    signalNodeStyle(node) {
      return {
        left: `${node.x / 14}%`,
        top: `${node.y / 7.8}%`,
      };
    },
    signalNodeState(node) {
      if (!this.activeSignalRoute) return "";
      if (this.activeSignalRoute.from.id === node.id) return "signal-source";
      if (this.activeSignalRoute.to.id === node.id) return "signal-target";
      return "";
    },
    signalEdgeState(edge) {
      return this.activeSignalRoute?.edgeId === edge.id ? "signal-edge-active" : "";
    },
    signalNodeForActor(actor) {
      const value = String(actor || "").toLowerCase();
      const aliases = [
        { id: "dialogue", terms: ["dialogue", "codex", "meta evaluator", "對話框"] },
        { id: "observer", terms: ["observer", "vue", "觀測"] },
        { id: "teacher", terms: ["teacher", "ollama", "qwen", "老師"] },
        { id: "learner", terms: ["learner", "pytorch", "學習器"] },
        { id: "arbiter", terms: ["agent", "arbiter", "decision", "決策"] },
        { id: "validators", terms: ["validator", "diagnosis", "驗證"] },
        { id: "unity", terms: ["unity", "deployment review", "部署"] },
        { id: "human", terms: ["human", "人工"] },
        { id: "scaffold", terms: ["scaffold", "bridge"] },
        { id: "governance", terms: ["governance", "policy", "治理"] },
        { id: "artifact", terms: ["artifact", "contract", "report", "event bus", "證據"] },
        { id: "sfm", terms: ["sfm", "colmap"] },
        { id: "train", terms: ["train", "mcmc", "3dgs"] },
        { id: "export", terms: ["export", "ply"] },
        { id: "l0", terms: ["production", "runtime", "l0", "frame", "生產"] },
      ];
      const match = aliases.find((entry) => entry.terms.some((term) => value.includes(term)));
      return this.signalNodes.find((node) => node.id === match?.id) || null;
    },
    directSignalPath(from, to) {
      const midpointY = Math.max(45, Math.min(from.y, to.y) - 70);
      return `M${from.x} ${from.y} C${from.x} ${midpointY} ${to.x} ${midpointY} ${to.x} ${to.y}`;
    },
    stageNode(stage, step, title, subtitle) {
      const event = this.eventFor(stage);
      const decision = this.decisionFor(stage);
      const exists = Boolean(event.run_id);
      const completed = event.status === "completed";
      const failed = exists && !completed && ["failed", "error", "blocked"].includes(event.status);
      const status = completed ? (decision.requires_human_review ? "review" : "ready") : failed ? "blocked" : "missing";
      return {
        id: stage,
        step,
        kicker: stage === "sfm" ? "幾何重建" : stage === "train" ? "模型訓練" : "部署轉換",
        title,
        subtitle,
        status,
        stateLabel: completed ? (decision.requires_human_review ? "完成，待審查" : "完成") : failed ? "失敗或阻塞" : "未執行",
        detail: exists
          ? `正式 event 狀態為 ${event.status || "unknown"}；正式 decision 為 ${decision.decision || "尚未產生"}。`
          : `尚未找到 ${stage} 的正式 latest event。`,
        evidence: event.run_root || `latest_${stage}_complete.json missing`,
        boundary: "生產層執行節點；結果必須寫入正式 event / report。",
      };
    },
    formatAge(seconds) {
      if (typeof seconds !== "number") return "-";
      if (seconds < 60) return `${Math.round(seconds)}s`;
      if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
      if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
      return `${Math.round(seconds / 86400)}d`;
    },
    shortTime(value) {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleTimeString("zh-TW", { hour12: false });
    },
  },
}).mount("#app");
