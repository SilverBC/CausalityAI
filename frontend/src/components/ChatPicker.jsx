/* Chat switcher for the Investigator panel: a styled listbox in place of a native <select>.
   Keyboard: Enter / Space / ↓ / ↑ open; ↑ ↓ move; Home / End jump; Enter picks; Esc closes only the list; Tab closes. */
import { useEffect, useId, useRef, useState } from "react";
import { ago } from "../format.js";
import { CheckIcon, ChevronIcon } from "./icons.jsx";

/** chats: [{id, title, updated}]; value: the current chat id, or null for a new chat */
export default function ChatPicker({ chats, value, onChange }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const box = useRef(null), btn = useRef(null), list = useRef(null);
  const id = useId();
  const options = [{ id: null, title: "New chat" }, ...chats];
  const current = options.find((o) => o.id === (value || null)) || options[0];

  useEffect(() => {
    if (!open) return;
    const close = (e) => { if (!box.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  useEffect(() => { if (open) list.current?.querySelector(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" }); }, [open, active]);

  const show = () => { setActive(Math.max(0, options.indexOf(current))); setOpen(true); };
  const choose = (o) => { setOpen(false); btn.current?.focus(); if ((o.id || null) !== (value || null)) onChange(o.id); };

  function onKey(e) {
    if (!open) {
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(e.key)) { e.preventDefault(); show(); }
      return;
    }
    const last = options.length - 1;
    const moves = { ArrowDown: (a) => Math.min(last, a + 1), ArrowUp: (a) => Math.max(0, a - 1), Home: () => 0, End: () => last };
    if (moves[e.key]) { e.preventDefault(); setActive(moves[e.key]); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(options[active]); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setOpen(false); } // not the whole panel
    else if (e.key === "Tab") setOpen(false);
  }

  return (
    <div className="picker" ref={box}>
      <button ref={btn} type="button" className={`picker-btn${open ? " open" : ""}`} aria-haspopup="listbox" aria-expanded={open}
        aria-controls={`${id}-list`} aria-activedescendant={open ? `${id}-${active}` : undefined}
        aria-label={`Chat: ${current.title}`} onClick={() => (open ? setOpen(false) : show())} onKeyDown={onKey}>
        <span className="picker-value" title={current.id ? current.title : undefined}>
          {current.id ? current.title : <span className="muted">New chat</span>}
        </span>
        <ChevronIcon />
      </button>
      {open && (
        <ul className="picker-list" role="listbox" id={`${id}-list`} ref={list} aria-label="Chats">
          {options.map((o, i) => {
            const selected = (o.id || null) === (current.id || null);
            return (
              <li key={o.id || "new"} id={`${id}-${i}`} data-i={i} role="option" aria-selected={selected}
                className={`picker-opt${i === active ? " active" : ""}${selected ? " selected" : ""}${o.id ? "" : " new"}`}
                onPointerEnter={() => setActive(i)} onClick={() => choose(o)}>
                <span className="picker-check">{selected && <CheckIcon />}</span>
                <span className="picker-title" title={o.id ? o.title : undefined}>{o.id ? o.title : "+ New chat"}</span>
                {o.id && <span className="picker-time small muted">{ago(o.updated)}</span>}
              </li>
            );
          })}
          {chats.length === 0 && <li className="picker-empty small muted" role="presentation">No saved chats for this analysis yet.</li>}
        </ul>
      )}
    </div>
  );
}
