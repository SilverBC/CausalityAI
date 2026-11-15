/* Light/dark mode. Until the user picks one, the app follows the device; a pick is stored in this browser and applied
   as <html data-theme="light|dark"> (index.html applies it before the first paint). The CSS tokens key off both. */
import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { MoonIcon, SunIcon } from "./components/icons.jsx";

const KEY = "causal-audit:theme";
export const THEME_EVENT = "causal-audit:theme-changed";
const mq = () => window.matchMedia("(prefers-color-scheme: dark)");

function stored() {
  try { const v = localStorage.getItem(KEY); return v === "light" || v === "dark" ? v : null; } catch { return null; }
}
const effective = () => stored() || (mq().matches ? "dark" : "light");

function apply(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(KEY, t); } catch { /* storage blocked: applies to this page only */ }
  window.dispatchEvent(new CustomEvent(THEME_EVENT, { detail: t }));
}

/** Switch with an animation: a circle of the new theme grows from `origin` (the toggle) where view transitions are
    supported, a short colour fade otherwise, and none for people who prefer reduced motion. */
export function setTheme(t, origin = null) {
  const root = document.documentElement;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return apply(t);
  if (!document.startViewTransition) {
    root.classList.add("theme-fade");
    apply(t);
    clearTimeout(window.__caThemeFade);
    window.__caThemeFade = setTimeout(() => root.classList.remove("theme-fade"), 450);
    return;
  }
  const x = origin?.x ?? window.innerWidth - 40, y = origin?.y ?? 30;
  const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
  // Rendering is paused inside the update callback, so update synchronously (flushSync). Waiting for animation
  // frames in there stalls the transition (a v1 bug).
  const vt = document.startViewTransition(() => flushSync(() => apply(t)));
  vt.ready.then(() => {
    root.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
      { duration: 560, easing: "cubic-bezier(.4, 0, .2, 1)", pseudoElement: "::view-transition-new(root)" },
    );
  }).catch(() => {});
}

export function useTheme() {
  const [theme, set] = useState(effective);
  useEffect(() => {
    const onSystem = () => { if (!stored()) set(effective()); };
    const onPick = (e) => set(e.detail);
    const m = mq();
    m.addEventListener("change", onSystem);
    window.addEventListener(THEME_EVENT, onPick);
    return () => { m.removeEventListener("change", onSystem); window.removeEventListener(THEME_EVENT, onPick); };
  }, []);
  return theme;
}

/** Sun/moon button. It shows the mode you would switch to. */
export function ThemeToggle() {
  const theme = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  const onClick = (e) => {
    const b = e.currentTarget.getBoundingClientRect();
    setTheme(next, { x: b.left + b.width / 2, y: b.top + b.height / 2 });
  };
  return (
    <button type="button" className="icon-btn theme-toggle" onClick={onClick} aria-label={`Switch to ${next} mode`} title={`Switch to ${next} mode`}>
      {theme === "dark" ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}
