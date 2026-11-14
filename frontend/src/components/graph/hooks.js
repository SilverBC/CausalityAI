/* Data hooks for the Investigation page: the data-search depth and the literature investigations. */
import { useCallback, useEffect, useState } from "react";
import {
  INVESTIGATIONS_EVENT, cancelQueuedInvestigations, deleteInvestigation, fetchInfluence, fetchInvestigations,
  investigationsChanged, startInvestigation, startInvestigations,
} from "../../api.js";
import { confirmAction } from "../ConfirmDialog.jsx";

/** The influence graph at a data-search depth; the analysis's own depth is already in the result. */
export function useDepthGraph(base, jobId) {
  const defaultDepth = base?.default_depth ?? 1;
  const dataMax = base?.max_depth ?? defaultDepth; // beyond this the slider only reveals literature nodes
  const [depth, setDepth] = useState(defaultDepth);
  const [inf, setInf] = useState(base);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { setDepth(defaultDepth); setInf(base); setError(""); }, [base, defaultDepth]);

  const dataDepth = Math.min(depth, dataMax);
  useEffect(() => {
    if (!base || dataDepth === defaultDepth) { setInf(base); setLoading(false); setError(""); return; }
    let live = true;
    setLoading(true);
    const t = setTimeout(() => { // debounce slider drags
      fetchInfluence(jobId, dataDepth)
        .then((g) => { if (live) { setInf(g); setError(""); } })
        .catch((e) => live && setError(e.message))
        .finally(() => live && setLoading(false));
    }, 250);
    return () => { live = false; clearTimeout(t); };
  }, [dataDepth, base, defaultDepth, jobId]);

  return { inf, depth, setDepth, defaultDepth, dataMax, loading, error };
}

export function depthStatus(inf, depth, defaultDepth, dataMax) {
  if (!inf) return "";
  if (depth > dataMax) return `The data search stops at depth ${dataMax}; deeper levels show only literature variables.`;
  if (depth < defaultDepth) return `Showing variables within ${depth} step${depth > 1 ? "s" : ""} of the outcome.`;
  if (depth === defaultDepth) return "The graph from the analysis. Slide right to search the dataset for causes further upstream.";
  const nv = inf.nodes.filter((n) => n.found).length;
  const nc = inf.nodes.filter((n) => n.found && n.role === "confounder").length;
  const ne = inf.edges.filter((e) => e.found).length;
  if (!nv && !ne) return "No further upstream causes found in this dataset.";
  const parts = [`${nv} new variable${nv === 1 ? "" : "s"}`];
  if (nc) parts.push(`${nc} of them confounder${nc === 1 ? "" : "s"}`);
  parts.push(`${ne} new link${ne === 1 ? "" : "s"}`);
  return `Found ${parts.join(", ")} in the data.${inf.exhausted ? " Nothing further upstream." : ""}`;
}

/** Literature investigations of one analysis: list, start, cancel, delete. Polls while any is queued or running. */
export function useInvestigations(jobId) {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    if (!jobId) return;
    try { setItems(await fetchInvestigations(jobId)); } catch { /* transient: the next poll retries */ }
  }, [jobId]);

  useEffect(() => { setItems([]); setError(""); refresh(); }, [refresh]);
  useEffect(() => { // started or deleted somewhere else (the sidebar, another tab of this page)
    const on = (e) => e.detail?.jobId === jobId && refresh();
    window.addEventListener(INVESTIGATIONS_EVENT, on);
    return () => window.removeEventListener(INVESTIGATIONS_EVENT, on);
  }, [jobId, refresh]);

  const busy = items.some((i) => i.status === "running" || i.status === "queued");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [busy, refresh]);

  const act = async (fn) => {
    setError("");
    try {
      const out = await fn();
      await refresh();
      investigationsChanged(jobId);
      return out;
    } catch (e) {
      setError(e.message);
      return null;
    }
  };

  const start = (node) => act(() => startInvestigation(jobId, node));
  const startMany = (nodes) => act(async () => {
    const r = await startInvestigations(jobId, nodes);
    if (r.skipped?.length) setError(`${r.skipped.length} could not be queued: the queue is full (40 at most). Cancel some or wait.`);
    return r;
  });
  const cancel = (node) => act(() => deleteInvestigation(jobId, node)); // nothing has run yet: no question asked
  const cancelQueued = () => act(() => cancelQueuedInvestigations(jobId));
  async function remove(node, label) {
    if (!(await confirmAction({
      title: `Delete the investigation of “${label || node}”?`, confirmLabel: "Delete investigation",
      message: "Its hypothetical findings are removed from the graph. You can investigate this variable again later.",
    }))) return false;
    return (await act(() => deleteInvestigation(jobId, node))) !== null;
  }

  return { items, error, refresh, start, startMany, cancel, cancelQueued, remove };
}
