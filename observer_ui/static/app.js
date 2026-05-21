const { createApp } = Vue;

createApp({
  data() {
    return {
      snapshot: null,
      error: null,
      lastLoadedAt: null,
      timer: null,
      stages: [
        { key: "sfm", label: "1A" },
        { key: "train", label: "1B" },
        { key: "export", label: "2" },
      ],
    };
  },
  computed: {
    artifacts() {
      return this.snapshot?.artifacts || [];
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
    bestLpips() {
      const value = this.snapshot?.mcmc?.best_lpips_run?.lpips;
      return typeof value === "number" ? value.toFixed(6) : "-";
    },
    bestRun() {
      return this.snapshot?.mcmc?.best_lpips_run?.run_name || "-";
    },
    deploymentVerdict() {
      const review = this.snapshot?.deployment_review || {};
      return review.unity_result || review.deployment_verdict || "not reviewed";
    },
    deploymentNote() {
      const review = this.snapshot?.deployment_review || {};
      return review.human_observation || review.note || review.reason || "deployment review is read-only";
    },
  },
  mounted() {
    this.loadSnapshot();
    this.timer = setInterval(this.loadSnapshot, 2500);
  },
  unmounted() {
    if (this.timer) clearInterval(this.timer);
  },
  methods: {
    async loadSnapshot() {
      try {
        const response = await fetch("/api/snapshot", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.snapshot = await response.json();
        this.error = null;
        this.lastLoadedAt = Date.now();
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
    percent(value) {
      return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
    },
    formatAge(seconds) {
      if (typeof seconds !== "number") return "-";
      if (seconds < 60) return `${Math.round(seconds)}s`;
      if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
      if (seconds < 86400) return `${Math.round(seconds / 3600)}h`;
      return `${Math.round(seconds / 86400)}d`;
    },
  },
}).mount("#app");
