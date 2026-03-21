import type {
  ChatMessage,
  TravelChatResponse,
  StructuredChatPayload,
} from "@/types";

function resolveApiUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (configured) return configured;
  if (typeof window !== "undefined") {
    const { origin, hostname } = window.location;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return "http://localhost:8000";
    }
    // In deployed environments, default to same-origin if env is missing.
    return origin;
  }
  return "http://localhost:8000";
}

const API_URL = resolveApiUrl();

export async function sendTravelChat(messages: ChatMessage[]): Promise<TravelChatResponse> {
  const res = await fetch(`${API_URL}/api/travel-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Chat API error");
  }
  return res.json();
}

export async function streamTravelChat(
  messages: ChatMessage[],
  onDelta: (delta: string) => void,
  onDone?: (payload: { message: string; structured?: StructuredChatPayload }) => void,
  sessionId?: string
): Promise<void> {
  const res = await fetch(`${API_URL}/api/travel-chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Chat stream API error");
  }

  if (!res.body) {
    throw new Error("Streaming response body is missing");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      const line = event
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!line) continue;
      const payload = line.slice(6);

      try {
        const parsed = JSON.parse(payload) as {
          delta?: string;
          done?: boolean;
          error?: string;
          message?: string;
          structured?: StructuredChatPayload;
        };
        if (parsed.error) {
          throw new Error(parsed.error);
        }
        if (parsed.delta) {
          onDelta(parsed.delta);
        }
        if (parsed.done && onDone && parsed.message) {
          onDone({ message: parsed.message, structured: parsed.structured });
        }
      } catch (error) {
        throw error instanceof Error
          ? error
          : new Error("Invalid stream payload received");
      }
    }
  }
}
