/* Explicit cloud preparation, followed by local-only projection. */
window.BookforgeAnticipation = {
  async init({sessionId, visualStyle, currentProjection, onShow}) {
    const get = (name) => document.getElementById(`nextPage${name}`);
    const text = get("Text");
    const prepare = get("Prepare");
    const show = get("Show");
    const discard = get("Discard");
    const warm = get("Warm");
    const status = get("Status");
    const metrics = get("Metrics");
    let enabled = false;
    let epoch = null;
    let page = null;
    let busy = false;
    let shown = false;
    let timer = null;
    let pollDeadline = 0;

    async function api(path, method = "GET", body) {
      const response = await fetch(`/v1/prepared-projections${path}`, {
        method,
        headers: body ? {"Content-Type": "application/json"} : {},
        body: body ? JSON.stringify(body) : undefined,
      });
      if (response.status === 204) return null;
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === "string"
        ? result.detail : "The request could not be completed.");
      return result;
    }

    function controls() {
      text.disabled = busy || Boolean(page);
      prepare.disabled = !enabled || busy || (!page && text.value.trim().length < 3)
        || page?.state === "staged";
      prepare.textContent = page ? "Check preparation" : "Prepare next scene";
      show.disabled = busy || page?.state !== "staged" || shown;
      discard.disabled = busy || !page;
      warm.disabled = !enabled || busy;
    }

    function render(snapshot) {
      page = snapshot;
      status.textContent = snapshot.detail;
      metrics.textContent = `Plan ${(snapshot.planning_ms / 1000).toFixed(2)} s · `
        + (snapshot.cache_hit ? "Cloud cache hit" :
          `Render ${(snapshot.render_ms / 1000).toFixed(2)} s · Review ${(snapshot.critic_ms / 1000).toFixed(2)} s`)
        + (snapshot.assets_verified ? " · Assets verified locally" : "");
      controls();
    }

    async function check() {
      if (page.state === "preparing") {
        render(await api(`/${page.prepared_id}?wait_seconds=20`));
      }
      if (page.state === "approved") {
        status.textContent = "Downloading and verifying the approved artwork and depth…";
        render(await api(`/${page.prepared_id}/stage`, "POST"));
      }
      if (page.state === "preparing" && Date.now() < pollDeadline) {
        timer = window.setTimeout(() => run(check), 100);
      } else if (page.state === "preparing") {
        status.textContent = "Still preparing. Use Check preparation to resume watching; it will not start another generation.";
      }
    }

    async function run(action) {
      if (busy) return;
      window.clearTimeout(timer);
      busy = true;
      controls();
      try { await action(); }
      catch (error) { status.textContent = error.message; }
      finally { busy = false; controls(); }
    }

    prepare.addEventListener("click", () => run(async () => {
      pollDeadline = Date.now() + 4 * 60 * 1000;
      if (!page) {
        status.textContent = "Planning locally, then preparing in GKE. Current projection unchanged…";
        render(await api("", "POST", {
          text: text.value.trim(), visual_style: visualStyle(), session_id: sessionId,
        }));
      }
      await check();
    }));

    show.addEventListener("click", () => run(async () => {
      const current = currentProjection();
      const pointer = await api(":activate", "POST", {
        prepared_id: page.prepared_id,
        expected_server_instance_id: current.server_instance_id || epoch,
        expected_session_revision: current.session_revision || 0,
      });
      onShow(pointer);
      shown = true;
      status.textContent = "Showing the prepared scene. No cloud generation was needed for this switch.";
    }));

    discard.addEventListener("click", () => run(async () => {
      await api(`/${page.prepared_id}`, "DELETE");
      page = null;
      shown = false;
      metrics.textContent = "";
      status.textContent = "Prepared page discarded. The projection is unchanged.";
    }));

    warm.addEventListener("click", () => run(async () => {
      status.textContent = "Warming the cloud path. This can take several minutes and uses credits…";
      const result = await api(":prewarm", "POST", {
        authorization: "I_UNDERSTAND_THIS_MAY_WAKE_A_BILLABLE_GPU",
      });
      status.textContent = result.detail || "Cloud path is warm.";
    }));
    text.addEventListener("input", controls);
    window.addEventListener("beforeunload", () => window.clearTimeout(timer));
    await run(async () => {
      const runtime = await api("/runtime");
      enabled = runtime.enabled;
      epoch = runtime.server_instance_id;
      status.textContent = enabled
        ? "Ready to prepare a next page. No cloud request has been made by this panel."
        : "Next-page preparation is not connected on this device yet. Regular generation is unchanged.";
    });
  },
};
