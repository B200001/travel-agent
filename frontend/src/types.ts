export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export type StructuredBlock =
  | { type: "heading"; text: string }
  | { type: "fact"; label: string; value: string }
  | { type: "paragraph"; lines: string[] }
  | { type: "list"; items: string[] };

export interface StructuredChatPayload {
  blocks: StructuredBlock[];
}

export interface TravelChatResponse {
  message: string;
  structured?: StructuredChatPayload;
}
