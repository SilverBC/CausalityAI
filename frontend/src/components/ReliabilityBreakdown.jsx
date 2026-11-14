/* "How is this scored?" dialog for a hypothetical variable's Reliability Score. */
import { useEffect, useRef } from "react";

const n2 = (v) => (v == null ? "–" : Number(v).toFixed(2));
const n3 = (v) => (v == null ? "–" : Number(v).toFixed(3));

export default function ReliabilityBreakdown({ factor, label, open, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const b = factor?.reliability_breakdown;
  const sources = factor?.sources || [];
  return (
    <dialog ref={ref} className="dialog rel-dialog" aria-labelledby="rel-title" onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}>
      {factor && (
        <div className="dialog-body">
          <div className="card-title-row">
            <div>
              <h2 id="rel-title">Reliability Score: {factor.reliability}/100</h2>
              <div className="small muted">{label}</div>
            </div>
            <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
          </div>
          <p className="small muted">
            Each source is looked up in OpenAlex, an open index of scholarly papers, for its citation count and publication
            type. The score says how established the cited research is. It does not say whether this factor actually
            confounds your data.
          </p>

          <h3>1 · Score each source</h3>
          <p className="small muted formula">
            source score = 0.6 × citation score + 0.4 × venue score<br />
            citation score = min(1, log₁₀(citations + 1) / 3), so 1,000+ citations = 1.0 · venue: journal 1.0,
            book/conference 0.8, working paper 0.6, report 0.5, web page 0.25
          </p>
          <div className="tbl-wrap">
            <table className="rel-table">
              <thead>
                <tr><th>Source</th><th className="num">Citations</th><th className="num">Citation score</th><th>Venue</th>
                  <th className="num">Venue score</th><th className="num">Source score</th></tr>
              </thead>
              <tbody>
                {sources.map((s, i) => (
                  <tr key={i} className={i === 0 ? "best" : ""}>
                    <td>
                      <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>
                      <div className="small muted">{s.verified ? "✓ found in OpenAlex" : "✗ not found in OpenAlex: no citation credit"}{i === 0 && " · strongest"}</div>
                    </td>
                    <td className="num">{s.citations != null ? s.citations.toLocaleString() : "–"}</td>
                    <td className="num">{n2(s.cite_score)}</td>
                    <td className="small">{s.venue_kind || "–"}{s.venue ? <div className="muted">{s.venue}</div> : null}</td>
                    <td className="num">{n2(s.venue_score)}</td>
                    <td className="num"><b>{n2(s.source_score)}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>2 · Combine into the variable's score</h3>
          {b ? (
            <dl className="dl">
              <dt>Strongest source</dt><dd>0.85 × {n3(b.best_source_score)} = {n3(b.best_weighted)}</dd>
              <dt>Corroboration</dt>
              <dd>{b.n_sources} source{b.n_sources === 1 ? "" : "s"} → +{n3(b.corroboration_bonus)} <span className="muted">(+0.075 per extra source, up to 2 extra)</span></dd>
              <dt>Total</dt><dd>{n3(b.best_weighted)} + {n3(b.corroboration_bonus)} = {n3(b.raw)}</dd>
              <dt>Reliability Score</dt><dd>{n3(Math.min(1, b.raw))} × 100, rounded = <b>{factor.reliability}/100</b></dd>
            </dl>
          ) : <p className="small muted">No breakdown was stored for this result.</p>}
        </div>
      )}
    </dialog>
  );
}
