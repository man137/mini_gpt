import { useEffect, useRef, useState } from 'react'
import ChatWindow from './components/ChatWindow.jsx'
import SourcesPanel from './components/SourcesPanel.jsx'
import Message from './components/Message.jsx'
import { fetchHealth, streamChat } from './api.js'

/**
 * Top-level chat application state:
 *  - messages: [{role, content, sources?, error?}, ...]
 *  - streaming: true while tokens are arriving (drives the stop button)
 *  - health: last /health snapshot for the status pill
 */
export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hi! I am the mini-chatgpt public-health assistant. I answer from the ' +
        'documents in the knowledge base and cite my sources. Ask me something ' +
        '(e.g. "How long does the flu vaccine take to work?").',
    },
  ])
  const [streaming, setStreaming] = useState(false)
  const [useRag, setUseRag] = useState(true)
  const [health, setHealth] = useState({ status: 'checking', backend: '...', index_size: 0 })
  const abortRef = useRef(null)

  useEffect(() => {
    let cancelled = false
    fetchHealth().then((h) => { if (!cancelled) setHealth(h) })
    return () => { cancelled = true }
  }, [])

  const handleSend = async (text, { useRag: overrideRag } = {}) => {
    if (!text.trim() || streaming) return
    const activeRag = overrideRag !== undefined ? overrideRag : useRag
    const userTurn = { role: 'user', content: text }
    const assistantTurn = { role: 'assistant', content: '', sources: [] }
    // Optimistic UI: append both turns immediately; tokens fill the last one.
    const history = [...messages, userTurn]
    setMessages([...history, assistantTurn])
    setStreaming(true)

    abortRef.current = new AbortController()

    // Conversation history sent to the backend (excluding our greeting).
    const apiHistory = history
      .filter((m, i) => !(i === 0 && m.role === 'assistant'))
      .map(({ role, content }) => ({ role, content }))

    await streamChat(
      apiHistory,
      {
        onToken: (t) =>
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              content: copy[copy.length - 1].content + t,
            }
            return copy
          }),
        onSources: (sources) =>
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = { ...copy[copy.length - 1], sources }
            return copy
          }),
        onDone: () => setStreaming(false),
        onError: (detail) => {
          setMessages((prev) => {
            const copy = [...prev]
            copy[copy.length - 1] = {
              ...copy[copy.length - 1],
              error: detail,
              content: copy[copy.length - 1].content || '(no answer)',
            }
            return copy
          })
          setStreaming(false)
        },
      },
      { signal: abortRef.current.signal, useRag: activeRag },
    )
  }

  const handleStop = () => {
    if (abortRef.current) abortRef.current.abort()
    setStreaming(false)
  }

  const handleRetry = () => {
    // Re-send the last user message (drop the failed assistant turn).
    const lastUser = [...messages].reverse().find((m) => m.role === 'user')
    if (!lastUser) return
    let cut = messages.length - 1
    while (cut > 0 && messages[cut].role !== 'user') cut -= 1
    setMessages(messages.slice(0, cut))
    setTimeout(() => handleSend(lastUser.content), 0)
  }

  const handleClear = () => {
    setMessages([messages[0]]) // keep only the greeting
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1>mini-chatgpt</h1>
          <span className="subtitle">public-health assistant</span>
        </div>
        <div className="header-right">
          <label className="rag-toggle" title="When checked, answers strictly using ingested documents and cites sources. When unchecked, answers with general AI knowledge.">
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
            />
            <span>RAG Mode</span>
          </label>
          <span
            className={`pill ${health.status === 'ok' ? 'pill-ok' : 'pill-bad'}`}
            title={`backend: ${health.backend}`}
          >
            {health.status === 'ok' ? 'online' : 'offline'}
            {health.index_size > 0 ? ` · ${health.index_size} chunks` : ''}
          </span>
          <button className="ghost-btn" onClick={handleClear}>Clear</button>
        </div>
      </header>

      <main className="main">
        <ChatWindow
          messages={messages}
          streaming={streaming}
          onSend={handleSend}
          onStop={handleStop}
          onRetry={handleRetry}
        />
        <SourcesPanel messages={messages} />
      </main>

      <footer className="footer">
        Educational project - not medical advice. Answers cite
        <code> [source:chunk_id]</code> when grounded in the documents.
      </footer>
    </div>
  )
}
