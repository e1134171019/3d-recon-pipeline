const { createApp } = Vue;

createApp({
  data() {
    return {
      snapshot: null,
      catalog: null,
      error: null,
      lastLoadedAt: null,
      lastMetaEventId: null,
      topologySignature: null,
      pulseNodeId: null,
      selectedNodeId: "reference",
      selectedCatalogNodeId: "mission",
      selectedExperimentId: "formal_mcmc",
      activeView: "project",
      timer: null,
      navItems: [
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
      if (this.error) return "observer error";
      return this.isStale ? "heartbeat stale" : "observer live";
    },
    metaActivity() {
      return this.snapshot?.meta_activity || {};
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
    this.timer = window.setInterval(this.loadSnapshot, 2500);
  },
  unmounted() {
    if (this.timer) window.clearInterval(this.timer);
  },
  methods: {
    async loadCatalog() {
      try {
        const response = await fetch("/api/catalog", { cache: "no-store" });
        if (!response.ok) throw new Error(`Catalog HTTP ${response.status}`);
        this.catalog = await response.json();
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
