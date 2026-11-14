/* Accounts and guest mode.
   - Every visitor gets a random guest id (localStorage), sent with each /api request as X-Guest-Id, so a guest's work
     stays tied to this browser.
   - Signed-in users are identified by an HttpOnly session cookie; signing up or in moves this browser's guest work to
     the account.
   - <AuthProvider> exposes the user and opens the sign-in dialog from anywhere through useAuth(). */
import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import { fetchMe, logIn, logOut, signUp } from "./api.js";

const GUEST_KEY = "causal-audit:guest-id";
export const AUTH_EVENT = "causal-audit:auth-changed";

function newId() {
  return (crypto.randomUUID?.() || Math.random().toString(36).slice(2) + Date.now().toString(36)).replace(/[^A-Za-z0-9-]/g, "");
}

export function guestId() {
  try {
    let id = localStorage.getItem(GUEST_KEY);
    if (!id || !/^[A-Za-z0-9-]{8,64}$/.test(id)) {
      id = newId();
      localStorage.setItem(GUEST_KEY, id);
    }
    return id;
  } catch {
    window.__caGuest ||= newId(); // storage blocked: guest work lasts until this tab closes
    return window.__caGuest;
  }
}

/** Add the guest id to every same-origin /api request. Call once at startup. */
export function installGuestHeader() {
  const id = guestId();
  const orig = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = typeof input === "string" ? input : input.url;
    if (url.startsWith("/api/")) {
      const headers = new Headers(init.headers || (typeof input === "string" ? undefined : input.headers));
      headers.set("X-Guest-Id", id);
      init = { ...init, headers, credentials: "same-origin" };
    }
    return orig(input, init);
  };
}

const Ctx = createContext(null);
export const useAuth = () => useContext(Ctx);

function AuthDialog({ mode, setMode, onClose, onDone }) {
  const ref = useRef(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const signup = mode === "signup";

  useEffect(() => { ref.current?.showModal(); }, []);
  useEffect(() => setError(""), [mode]);

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onDone(await (signup ? signUp : logIn)(email, password));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <dialog ref={ref} className="dialog auth-dialog" aria-labelledby="auth-title" onClose={onClose}
      onClick={(e) => e.target === ref.current && ref.current.close()}>
      <form onSubmit={submit} className="dialog-body">
        <div className="card-title-row">
          <h2 id="auth-title">{signup ? "Create an account" : "Sign in"}</h2>
          <button type="button" className="icon-btn" onClick={() => ref.current.close()} aria-label="Close">✕</button>
        </div>
        <p className="small muted">
          {signup ? "Keep your analyses across devices. " : "Welcome back. "}
          Anything you did as a guest in this browser moves to your account.
        </p>
        <label className="field">
          <span className="lbl">Email</span>
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoFocus />
        </label>
        <label className="field">
          <span className="lbl">Password</span>
          <input type="password" autoComplete={signup ? "new-password" : "current-password"} required minLength={signup ? 8 : undefined}
            value={password} onChange={(e) => setPassword(e.target.value)} aria-describedby={signup ? "pw-hint" : undefined} />
          {signup && <span id="pw-hint" className="small muted">At least 8 characters.</span>}
        </label>
        {error && <p className="err small" role="alert">{error}</p>}
        <button className="btn primary" disabled={busy}>{busy ? "…" : signup ? "Create account" : "Sign in"}</button>
        <p className="small muted auth-switch">
          {signup ? "Already have an account? " : "New here? "}
          <button type="button" className="link-btn" onClick={() => setMode(signup ? "login" : "signup")}>
            {signup ? "Sign in" : "Create an account"}
          </button>
        </p>
      </form>
    </dialog>
  );
}

export function AuthProvider({ children }) {
  const [me, setMe] = useState(null); // {user, email_notifications}
  const [dialog, setDialog] = useState(null); // "login" | "signup" | null

  const refresh = useCallback(() => fetchMe().then(setMe).catch(() => setMe({ user: null, email_notifications: false })), []);
  useEffect(() => { refresh(); }, [refresh]);

  const changed = (next) => {
    setMe((prev) => ({ ...prev, ...next }));
    window.dispatchEvent(new CustomEvent(AUTH_EVENT, { detail: next }));
  };

  async function logout() {
    await logOut().catch(() => {});
    changed({ user: null });
  }

  const value = {
    ready: !!me, user: me?.user || null, emailNotifications: !!me?.email_notifications,
    openAuth: (mode = "signup") => setDialog(mode), logout,
  };
  return (
    <Ctx.Provider value={value}>
      {children}
      {dialog && <AuthDialog mode={dialog} setMode={setDialog} onClose={() => setDialog(null)}
        onDone={(next) => { changed(next); setDialog(null); }} />}
    </Ctx.Provider>
  );
}

/** Run `fn` whenever the user signs in or out (lists depend on who is asking). */
export function useOnAuthChange(fn) {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    const on = (e) => ref.current(e.detail);
    window.addEventListener(AUTH_EVENT, on);
    return () => window.removeEventListener(AUTH_EVENT, on);
  }, []);
}

/** "You're a guest" note with a create-account button; nothing for signed-in users. */
export function GuestNudge({ className = "", text = "You're using guest mode, so your work is saved only in this browser." }) {
  const { ready, user, openAuth } = useAuth();
  if (!ready || user) return null;
  return (
    <div className={`guest-nudge ${className}`} role="note">
      <span className="small">{text}</span>
      <button className="btn sm primary" onClick={() => openAuth("signup")}>Create an account to keep it</button>
    </div>
  );
}

/** "jane.doe@x.com" -> "JD", "alice@x.com" -> "A". */
export function initials(email = "") {
  const parts = email.split("@")[0].split(/[._+-]+/).filter((p) => /[a-z0-9]/i.test(p));
  return ((parts[0]?.[0] || "?") + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

/** Header account control: Sign in / Create account, or an initials avatar with a small menu. */
export function AccountButton() {
  const { ready, user, openAuth, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const box = useRef(null);
  useEffect(() => {
    if (!open) return;
    const close = (e) => { if (!box.current?.contains(e.target)) setOpen(false); };
    const esc = (e) => { if (e.key === "Escape") { setOpen(false); box.current?.querySelector(".acct-btn")?.focus(); } };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", esc); };
  }, [open]);
  if (!ready) return null;
  if (!user)
    return (
      <div className="acct">
        <button className="btn sm ghost" onClick={() => openAuth("login")}>Sign in</button>
        <button className="btn sm primary acct-create" onClick={() => openAuth("signup")}>Create account</button>
      </div>
    );
  return (
    <div className="acct" ref={box}>
      <button className="acct-btn" title={user.email} onClick={() => setOpen((v) => !v)} aria-expanded={open} aria-haspopup="menu"
        aria-label={`Account: ${user.email}`}>
        <span className="acct-avatar" aria-hidden="true">{initials(user.email)}</span>
        <span className="small acct-email">{user.email}</span>
      </button>
      {open && (
        <div className="acct-menu" role="menu">
          <div className="small muted acct-menu-email">{user.email}</div>
          <button role="menuitem" className="btn sm ghost" onClick={() => { setOpen(false); logout(); }}>Sign out</button>
        </div>
      )}
    </div>
  );
}
