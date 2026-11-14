/* Tiny hash router. Hash routes need no server-side fallback, so FastAPI's static mount keeps working.
     #/                         Analysis page
     #/?job=<id>                Analysis page showing a saved analysis (reload-safe; links in emails use it)
     #/investigate/<id>?node=   Investigation page (prompt 03)
     #/m...                     Phone app (prompt 05)
   The query string lives inside the hash, after "?". */
import { useEffect, useState } from "react";

export function parseHash(hash = window.location.hash) {
  const [path, qs = ""] = hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  const query = new URLSearchParams(qs);
  if (parts[0] === "m") return { page: "mobile", view: parts[1] || "home", id: parts[2] || null, query };
  if (parts[0] === "investigate") return { page: "investigate", jobId: parts[1] || null, query };
  return { page: "analysis", jobId: query.get("job"), query };
}

export function useRoute() {
  const [route, setRoute] = useState(parseHash);
  useEffect(() => {
    const onChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export const go = (hash) => { if (window.location.hash !== hash) window.location.hash = hash; };
