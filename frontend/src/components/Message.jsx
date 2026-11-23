/**
 * One chat bubble. Renders:
 *  - the text (with a blinking cursor while this message is streaming);
 *  - an inline error banner if the turn failed (with a retry button);
 *  - nothing else - sources live in the SourcesPanel.
 */
export default function Message({ message, streaming, onRetry }) {
  const isUser = message.role === 'user'

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      <div className="bubble">
        <span className="bubble-text">
          {message.content}
          {streaming && <span className="cursor">▍</span>}
        </span>
        {message.error && (
          <div className="error-banner">
            <span>⚠ {message.error}</span>
            {onRetry && (
              <button className="retry-btn" onClick={onRetry}>
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
