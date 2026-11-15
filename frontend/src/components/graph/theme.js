/* Shared look of the graph views: role colours and labels, and canvas colours read from the CSS tokens. */
import { useEffect, useState } from "react";
import { THEME_EVENT } from "../../theme.jsx";

export const ROLE_VAR = {
  treatment: "--treat", outcome: "--ctrl", confounder: "--warn", instrument: "--instr", running: "--instr",
  post_treatment: "--fail", latent: "--latent", fixed_effect: "--muted", group: "--muted", upstream: "--accent",
  hypothetical: "--hypo", literature_data: "--muted",
};

export const ROLE_LABEL = {
  treatment: "treatment", outcome: "outcome", confounder: "confounder (assumed)", instrument: "instrument",
  running: "running variable", post_treatment: "measured after treatment", latent: "unmeasured (assumed)",
  fixed_effect: "fixed effect", group: "other covariates (grouped)", variable: "variable",
  upstream: "upstream cause (data search)",
  hypothetical: "hypothetical (literature, not in your data)", literature_data: "in your data, suggested by the literature",
};

/** Resolve the CSS colour tokens the canvas needs; re-resolve when the theme changes. */
export function useThemeColors() {
  const read = () => {
    const cs = getComputedStyle(document.documentElement);
    const v = (name) => cs.getPropertyValue(name).trim();
    return {
      ink: v("--ink"), muted: v("--muted"), line: v("--line"), panel: v("--panel"), node: v("--muted"),
      pos: v("--pass"), neg: v("--fail"), accent: v("--accent"), neutral: v("--ink2"), hypo: v("--hypo"),
      role: Object.fromEntries(Object.entries(ROLE_VAR).map(([k, name]) => [k, v(name)])),
    };
  };
  const [colors, setColors] = useState(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    // after the theme attribute changes, styles recompute on the next frame
    const onChange = () => requestAnimationFrame(() => setColors(read()));
    mq.addEventListener("change", onChange);
    window.addEventListener(THEME_EVENT, onChange);
    return () => { mq.removeEventListener("change", onChange); window.removeEventListener(THEME_EVENT, onChange); };
  }, []);
  return colors;
}

export function useWidth(ref) {
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export function withAlpha(color, a) {
  if (/^#[0-9a-f]{6}$/i.test(color)) return color + Math.round(a * 255).toString(16).padStart(2, "0");
  return color;
}
