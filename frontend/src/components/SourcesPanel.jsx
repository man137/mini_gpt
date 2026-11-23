/**
 * Collapsible panel listing the retrieved sources of the LAST assistant
 * answer: document name, chunk id, similarity score and a snippet.
 * Collapsed by default so it never competes with the chat for space.
 */
export default function SourcesPanel({ messages }) {
  // Find the most recent assistant turn that carried sources.
  const lastWithSources = [...messages]
    .reverse()
    .find((m) => m.role === 'assistant' && m.sources && m.sources.length > 0)

  if (!lastWithSources) {
    return (
      <aside className="sources-panel">
        <h2>Sources</h2>
        <p className="sources-empty">
          No retrieved context yet. Ask a question and retrieved document
          chunks will appear here with their similarity scores.
        </p>
      </aside>
    )
  }

  return (
    <aside className="sources-panel">
      <details open>
        <summary>
          Sources <span className="sources-count">({lastWithSources.sources.length})</span>
        </summary>
        <ul className="sources-list">
          {lastWithSources.sources.map((s, i) => (
            <li key={i} className="source-card">
              <div className="source-head">
                <span className="source-name">{s.source}</span>
                <span className="source-score" title="cosine similarity">
                  {s.score != null ? s.score.toFixed(3) : ''}
                </span>
              </div>
              <code className="source-chunk">{s.chunk_id}</code>
              <p className="source-snippet">{s.snippet}…</p>
            </li>
          ))}
        </ul>
      </details>
    </aside>
  )
}
