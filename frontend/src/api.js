// Backend communication: SSE streaming via fetch + ReadableStream.
//
// Why fetch + ReadableStream instead of EventSource: EventSource only does
// GET requests with no JSON body or headers, but our /chat endpoint needs a
// POST with the conversation history. Streaming the POST response body
// manually gives us the same SSE parsing with full request control, plus
// AbortController for the stop button (EventSource has close(), but only
// before the response starts).

// Base URL of the backend. In dev, Vite proxies same-origin requests to
// localhost:8000 (see vite.config.js), so '' is correct there; production
// builds set VITE_API_URL at build time.
export const API_BASE = import.meta.env.VITE_API_URL || ''

/**
 * Stream a chat answer from POST /chat.
 *
 * @param {Array<{role: string, content: string}>} messages - conversation so far
 * @param {Object} callbacks - { onToken(text), onSources(list), onDone(info),
 *                               onError(detail) }
 * @param {Object} [options] - { signal (AbortController.signal), useRag }
 * @returns {Promise<void>} resolves when the stream ends
 */
export async function streamChat(messages, callbacks, options = {}) {
  const { onToken, onSources, onDone, onError } = callbacks
  const { signal, useRag = true } = options

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, use_rag: useRag }),
      signal,
    })

    if (!res.ok) {
      // Non-2xx: the backend returns JSON {"detail": "..."} for early
      // rejections (413 too long, 429 rate limit, 422 validation).
      let detail = `request failed with status ${res.status}`
      try {
        const body = await res.json()
        if (body && body.detail) detail = body.detail
      } catch { /* body was not JSON; keep the generic detail */ }
      onError(detail)
      return
    }

    // --- SSE parsing over the streamed body --------------------------------
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    // SSE frames are separated by a blank line; a frame looks like:
    //   event: token\n data: {"text": "..."}\n\n
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // Process every COMPLETE frame in the buffer.
      let sep
      while ((sep = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)

        let event = null
        let data = null
        for (const line of frame.split('\n')) {
          if (line.startsWith('event: ')) event = line.slice(7).trim()
          else if (line.startsWith('data: ')) {
            try { data = JSON.parse(line.slice(6)) }
            catch { data = line.slice(6) } // tolerate non-JSON payloads
          }
        }
        if (event === 'token' && data) onToken(data.text ?? '')
        else if (event === 'sources') onSources(Array.isArray(data) ? data : [])
        else if (event === 'done') onDone(data ?? {})
        else if (event === 'error') onError((data && data.detail) || 'stream error')
      }
    }
    // The stream ending without a "done" event is still a clean finish for
    // the UI (the backend always sends done, but be defensive).
    onDone({})
  } catch (err) {
    if (err.name === 'AbortError') {
      // User pressed stop: not an error, just finish the current turn.
      onDone({ aborted: true })
    } else {
      onError(err.message || 'network error')
    }
  }
}

/** GET /health - used by the header status pill. */
export async function fetchHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`)
    if (!res.ok) throw new Error(`status ${res.status}`)
    return await res.json()
  } catch (err) {
    return { status: 'offline', backend: 'unreachable', index_size: 0 }
  }
}

/** POST /ingest - upload documents and rebuild the RAG index. */
export async function uploadDocuments(fileList) {
  const form = new FormData()
  for (const f of fileList) form.append('files', f)
  const res = await fetch(`${API_BASE}/ingest`, { method: 'POST', body: form })
  if (!res.ok) {
    let detail = `upload failed (${res.status})`
    try { const b = await res.json(); if (b.detail) detail = b.detail } catch {}
    throw new Error(detail)
  }
  return res.json()
}
