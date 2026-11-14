/* Small formatting helpers. */

export const fmt = (v, d = 3) => {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e7 || (a < 1e-4 && v !== 0)) return v.toExponential(2);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: a >= 1e4 ? 0 : 1 });
  return (+v).toFixed(d).replace(/\.0+$|(\.\d*?)0+$/, "$1");
};

export function ago(t) {
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

export const MODE_LABEL = { audit: "Claim audit", explore: "Exploration", "claim-only": "Claim only" };
