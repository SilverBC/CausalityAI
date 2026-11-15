/* Display-only node names. A user can relabel any node; the name is kept per analysis in this browser's
   localStorage and is never sent to the backend or the language model. Everything else keeps using the original
   id and label. Works without storage too (names then last until the page is reloaded). */
import { useCallback, useEffect, useState } from "react";

const KEY = (jobId) => `causal-audit:names:${jobId}`;
const EVENT = "causal-audit:names-changed";

function load(jobId) {
  if (!jobId) return {};
  try { return JSON.parse(localStorage.getItem(KEY(jobId)) || "{}") || {}; } catch { return {}; }
}

export function useNodeNames(jobId) {
  const [names, setNames] = useState(() => load(jobId));

  useEffect(() => {
    setNames(load(jobId));
    const sync = (e) => e.detail?.jobId === jobId && setNames(e.detail.names);
    window.addEventListener(EVENT, sync);
    return () => window.removeEventListener(EVENT, sync);
  }, [jobId]);

  const rename = useCallback((id, name) => {
    setNames((prev) => {
      const next = { ...prev };
      const clean = (name || "").trim().slice(0, 60);
      if (clean) next[id] = clean; else delete next[id]; // empty = back to the original
      try { localStorage.setItem(KEY(jobId), JSON.stringify(next)); } catch { /* this page only */ }
      window.dispatchEvent(new CustomEvent(EVENT, { detail: { jobId, names: next } }));
      return next;
    });
  }, [jobId]);

  const nameOf = useCallback((id, original) => names[id] || original, [names]);
  return { names, rename, nameOf };
}
