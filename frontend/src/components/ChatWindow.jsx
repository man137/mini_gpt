import { useEffect, useRef, useState } from 'react'
import Message from './Message.jsx'

/**
 * The conversation pane + input box.
 * - auto-scrolls to the newest token while streaming (unless the user
 *   scrolled up manually - then we respect their position);
 * - Enter sends, Shift+Enter is a newline;
 * - shows a stop button while streaming, send button otherwise.
 */
export default function ChatWindow({ messages, streaming, onSend, onStop, onRetry }) {
  const [input, setInput] = useState('')
  const listRef = useRef(null)
  const pinnedToBottom = useRef(true)

  useEffect(() => {
    const el = listRef.current
    if (el && pinnedToBottom.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  const handleScroll = () => {
    const el = listRef.current
    if (!el) return
    // "pinned" = within 80px of the bottom
    pinnedToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }

  const submit = () => {
    if (!input.trim() || streaming) return
    onSend(input)
    setInput('')
  }

  return (
    <section className="chat-window">
      <div className="message-list" ref={listRef} onScroll={handleScroll}>
        {messages.map((m, i) => (
          <Message
            key={i}
            message={m}
            streaming={streaming && i === messages.length - 1}
            onRetry={onRetry}
          />
        ))}
      </div>

      <div className="composer">
        <textarea
          value={input}
          placeholder="Ask a public-health question..."
          rows={2}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        {streaming ? (
          <button className="stop-btn" onClick={onStop} title="Stop generating">
            ■ Stop
          </button>
        ) : (
          <button className="send-btn" onClick={submit} disabled={!input.trim()}>
            Send
          </button>
        )}
      </div>
    </section>
  )
}
