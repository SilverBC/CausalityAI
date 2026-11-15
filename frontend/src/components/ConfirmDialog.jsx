/* The app's own confirmation dialog (no native confirm/alert anywhere).
     if (!(await confirmAction({ title, message, confirmLabel: "Delete", tone: "danger" }))) return;
   Built on a modal <dialog>: focus stays inside and the page behind is inert. Cancel is focused first, so pressing
   Enter by accident never deletes anything. Esc and a click on the backdrop cancel; focus returns to the trigger.
   tone "danger": red trash icon + red button. tone "warn": amber triangle + primary button.
   On phones it is a bottom sheet with full-width buttons, Cancel last (see .confirm-dialog in styles.css). */
import { useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { TrashIcon, WarnIcon } from "./icons.jsx";

function Confirm({ title, message, confirmLabel, cancelLabel, tone, onDone }) {
  const ref = useRef(null);
  const cancelRef = useRef(null);
  const answered = useRef(false);
  const finish = (ok) => {
    if (answered.current) return;
    answered.current = true;
    ref.current?.close();
    onDone(ok);
  };
  useEffect(() => {
    ref.current?.showModal();
    cancelRef.current?.focus();
  }, []);

  return (
    <dialog ref={ref} className={`dialog confirm-dialog ${tone}`} aria-labelledby="confirm-title" aria-describedby="confirm-msg"
      onCancel={(e) => { e.preventDefault(); finish(false); }}
      onClick={(e) => e.target === ref.current && finish(false)}>
      <div className="confirm-body">
        <span className={`confirm-icon ${tone}`} aria-hidden="true">{tone === "warn" ? <WarnIcon size={20} /> : <TrashIcon size={20} />}</span>
        <div className="confirm-text">
          <h2 id="confirm-title">{title}</h2>
          {message && <p id="confirm-msg">{message}</p>}
        </div>
      </div>
      <div className="confirm-actions">
        <button ref={cancelRef} type="button" className="btn ghost" onClick={() => finish(false)}>{cancelLabel}</button>
        <button type="button" className={`btn ${tone === "warn" ? "primary" : "danger"}`} onClick={() => finish(true)}>{confirmLabel}</button>
      </div>
    </dialog>
  );
}

/** Ask the user to confirm. Resolves true when confirmed, false on Cancel, Esc or a backdrop click. */
export function confirmAction({ title, message = "", confirmLabel = "Confirm", cancelLabel = "Cancel", tone = "danger" } = {}) {
  return new Promise((resolve) => {
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    const trigger = document.activeElement;
    const done = (ok) => {
      resolve(ok);
      setTimeout(() => { root.unmount(); host.remove(); if (trigger?.isConnected) trigger.focus?.(); }, 0);
    };
    root.render(<Confirm {...{ title, message, confirmLabel, cancelLabel, tone }} onDone={done} />);
  });
}

/** "Its 2 investigations and 1 chat will be deleted too." (or "") */
export function whatGoesWith({ investigations = 0, chats = 0 }) {
  const parts = [investigations && `${investigations} investigation${investigations === 1 ? "" : "s"}`,
    chats && `${chats} chat${chats === 1 ? "" : "s"}`].filter(Boolean);
  return parts.length ? `Its ${parts.join(" and ")} will be deleted too. ` : "";
}
