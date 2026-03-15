import type { TripRequest, TravelPlanResult, ChatMessage, TravelChatResponse } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function sendTravelChat(messages: ChatMessage[]): Promise<TravelChatResponse> {
  const res = await fetch(`${API_URL}/api/travel-chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "Chat API error");
  }
  return res.json();
}

export async function planTrip(data: TripRequest): Promise<TravelPlanResult> {
  const res = await fetch(`${API_URL}/api/plan-trip`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || res.statusText || "API error");
  }

  return res.json();
}
