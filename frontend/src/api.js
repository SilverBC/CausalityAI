/* Thin wrappers around the FastAPI endpoints. Errors carry the server's plain-words message. */

async function parse(r) {
  let data = null;
  try { data = await r.json(); } catch { /* empty or non-JSON body */ }
  if (!r.ok) {
    const d = data?.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : null;
    throw Object.assign(new Error(msg || `The server answered ${r.status} ${r.statusText}.`), { status: r.status });
  }
  return data;
}

async function request(url, init) {
  let r;
  try {
    r = await fetch(url, init);
  } catch {
    throw new Error("The server could not be reached. Is it running?");
  }
  return parse(r);
}

const sendJSON = (url, method, body) =>
  request(url, { method, headers: { "Content-Type": "application/json" }, body: body && JSON.stringify(body) });

export const fetchHealth = () => request("/api/health");
export const fetchSamples = () => request("/api/samples");
export const fetchJob = (id) => request(`/api/jobs/${encodeURIComponent(id)}`);
export const fetchDatasets = () => request("/api/datasets");
export const fetchInfluence = (id, depth) => request(`/api/jobs/${encodeURIComponent(id)}/influence?depth=${depth}`);

export async function fetchSampleFile(sample) {
  const r = await fetch(`/api/samples/${encodeURIComponent(sample.id)}`);
  if (!r.ok) throw new Error("Could not load the sample.");
  return new File([await r.blob()], sample.filename || `${sample.id}.csv`, { type: "text/csv" });
}

export function fetchColumns(file) {
  const fd = new FormData();
  fd.append("file", file);
  return request("/api/columns", { method: "POST", body: fd });
}

/** Role overrides without empty values; randomized, post_treatment and cutoff keep their types. */
export function cleanRoles(roles) {
  const out = {};
  Object.entries(roles || {}).forEach(([k, v]) => {
    if (k === "randomized") { if (v) out.randomized = true; return; }
    if (k === "post_treatment") { if (v?.length) out.post_treatment = v; return; }
    if (k === "cutoff") { if (v !== "" && v != null && isFinite(Number(v))) out.cutoff = Number(v); return; }
    if (typeof v === "string" && v.trim()) out[k] = v.trim();
  });
  return out;
}

export function startAnalysis({ file, claim, design = "", roles = {}, online = false, notify = false }) {
  const fd = new FormData();
  if (file) fd.append("file", file);
  fd.append("claim", (claim || "").trim());
  fd.append("design", design || "");
  fd.append("roles", JSON.stringify(cleanRoles(roles)));
  fd.append("online_literature", online ? "true" : "false");
  fd.append("notify", notify ? "1" : "0");
  return request("/api/analyze", { method: "POST", body: fd }); // { job_id, mode }
}

export const answerQuestions = (id, { confirm = {}, design = null }) =>
  sendJSON(`/api/jobs/${encodeURIComponent(id)}/answer`, "POST", { confirm, design });

export const DATASETS_EVENT = "causal-audit:datasets-changed"; // fire after starting or finishing an analysis
export const datasetsChanged = () => window.dispatchEvent(new Event(DATASETS_EVENT));

// ---- literature investigations
const inv = (id) => `/api/jobs/${encodeURIComponent(id)}`;
export const fetchInvestigations = (id) => request(`${inv(id)}/investigations`);
export const startInvestigation = (id, n) =>
  sendJSON(`${inv(id)}/investigate`, "POST", { node: n.id, label: n.label, in_data: !n.hypothetical });
export const startInvestigations = (id, nodes) =>
  sendJSON(`${inv(id)}/investigate-batch`, "POST", { nodes: nodes.map((n) => ({ node: n.id, label: n.label, in_data: !n.hypothetical })) });
export const deleteInvestigation = (id, node) => sendJSON(`${inv(id)}/investigations/${encodeURIComponent(node)}`, "DELETE");
export const cancelQueuedInvestigations = (id) => sendJSON(`${inv(id)}/investigations/cancel-queued`, "POST");

export const INVESTIGATIONS_EVENT = "causal-audit:investigations-changed"; // detail: { jobId }
export const investigationsChanged = (jobId) => {
  window.dispatchEvent(new CustomEvent(INVESTIGATIONS_EVENT, { detail: { jobId } }));
  datasetsChanged();
};

// ---- deletion and accounts
export const deleteAnalysis = (id) => sendJSON(`/api/jobs/${encodeURIComponent(id)}`, "DELETE");
export const fetchMe = () => request("/api/auth/me");
export const signUp = (email, password) => sendJSON("/api/auth/signup", "POST", { email, password });
export const logIn = (email, password) => sendJSON("/api/auth/login", "POST", { email, password });
export const logOut = () => sendJSON("/api/auth/logout", "POST");

// ---- datasets (= analyses) and their chats with the Investigator
export const fetchDataset = (id) => request(`/api/datasets/${encodeURIComponent(id)}`);
export const createChat = (datasetId) => sendJSON(`/api/datasets/${encodeURIComponent(datasetId)}/chats`, "POST");
export const fetchChat = (id) => request(`/api/chats/${encodeURIComponent(id)}`);
export const deleteChat = (id) => sendJSON(`/api/chats/${encodeURIComponent(id)}`, "DELETE");
export const sendChatMessage = (id, text) => sendJSON(`/api/chats/${encodeURIComponent(id)}/messages`, "POST", { text });
