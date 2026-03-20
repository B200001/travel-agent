"use client";

import { useState, useRef, useEffect } from "react";
import { planTrip, streamTravelChat } from "@/lib/api";
import type { TripRequest, TravelPlanResult, ChatMessage, StructuredBlock } from "@/types";

const PREFERENCE_OPTIONS = {
  type: [
    { value: "leisure", label: "Leisure - Relaxed pace" },
    { value: "cultural", label: "Cultural - Heritage & History" },
    { value: "adventure", label: "Adventure - Seeking thrills" },
    { value: "party", label: "Party - Nightlife & fun" },
    { value: "family", label: "Family - With everyone" },
  ],
  pace: [
    { value: "relaxed", label: "Relaxed - Maximum comfort" },
    { value: "moderate", label: "Moderate - Balanced" },
    { value: "packed", label: "Packed - Full day busy" },
  ],
};

type Tab = "plan" | "chat";
const WELCOME_MESSAGE =
  "Hi! I'm your travel assistant. Tell me where you'd like to go, your budget, dates, or what kind of trip you're looking for - I'll help you plan it!";

type MessageBlock = StructuredBlock;
type UIChatMessage = ChatMessage & { blocks?: StructuredBlock[] };

function formatInline(text: string) {
  const tokens = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g).filter(Boolean);
  return tokens.map((token, idx) => {
    if (token.startsWith("**") && token.endsWith("**")) {
      return (
        <strong key={idx} className="font-semibold text-slate-900">
          {token.slice(2, -2)}
        </strong>
      );
    }
    if (token.startsWith("*") && token.endsWith("*")) {
      return (
        <em key={idx} className="italic">
          {token.slice(1, -1)}
        </em>
      );
    }
    return <span key={idx}>{token}</span>;
  });
}

function parseMessageBlocks(content: string): MessageBlock[] {
  const cleanLine = (line: string) =>
    line
      .replace(/\*{3,}/g, "**")
      .replace(/^\s*#+\s*#+\s*$/g, "")
      .replace(/^\s*#+\s*$/g, "")
      .replace(/^\s*-\s*Late\s*$/i, "")
      .replace(/^\s*Late\s*$/i, "")
      .replace(/\*\*(Morning|Afternoon|Evening|Lunch|Dinner|Full Day|Late Morning|Late Afternoon)\s*$/i, "$1")
      .replace(/\*\*(Morning|Afternoon|Evening|Lunch|Dinner|Full Day|Late Morning|Late Afternoon)\s*:\s*$/i, "$1:")
      .replace(/\s{2,}/g, " ")
      .trim();

  const normalized = content
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+/g, " ")
    .replace(/\s+(?=#{1,6}\s)/g, "\n\n")
    .replace(
      /\s+(?=(Friday,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+[A-Za-z ]+Opportunities))/gi,
      "\n\n"
    )
    .replace(
      /\s+(?=([A-Z][A-Za-z '&/-]{2,80},\s*(USA|Mexico|Canada|Australia|New Zealand|Iceland|Norway|Japan|South Africa)[^:]{0,80}:))/g,
      "\n\n"
    )
    .replace(
      /\.\s+(?=([A-Z][A-Za-z'&(). -]{2,80}:\s))/g,
      ".\n"
    )
    .replace(
      /\s+(?=([A-Za-z][A-Za-z'&()./ -]{2,80}\s\/\s[A-Za-z][A-Za-z'&()./ -]{2,80}:))/g,
      "\n"
    )
    .replace(
      /([A-Za-z][A-Za-z0-9 '&().-]{4,120},\s*[A-Za-z() .-]{2,80},\s*Australia)\s+(?=Operating Status for Today:)/g,
      "\n\n$1\n"
    )
    .replace(
      /\s+(?=(Wildlife Parks\s*&\s*Sanctuaries|National Parks\s*&\s*Gardens|Uncertainty\s*&\s*Recommendation)\b)/gi,
      "\n\n"
    )
    .replace(/\s+(?=(Operating Status for Today:|Viewing:))/gi, "\n")
    .replace(/\s+(?=(Your\s+\w+\s+Plan\b))/gi, "\n\n")
    .replace(/\s+(?=Day\s+\d+\b[^:]*:)/gi, "\n\n")
    .replace(/\s+(?=Budget Breakdown\b)/gi, "\n\n")
    .replace(/\s+(?=(Morning|Afternoon|Evening|Lunch|Dinner|Full Day|Note)\s*:)/gi, "\n")
    .replace(/\s+(?=(With your\s+₹|How does this sound\b|Let me know if))/gi, "\n\n")
    .replace(/\*\s*(Morning|Afternoon|Evening|Lunch|Dinner|Full Day)\s*:/gi, "\n- $1:")
    .replace(
      /\s+(?=(Flights|Accommodation|Food\s*&\s*Drink|Local Transportation|Activities\s*&\s*Sightseeing|Miscellaneous|Total Estimated Budget)\b)/gi,
      "\n"
    )
    .replace(/\n{3,}/g, "\n\n");

  const lines = normalized.split("\n").map((line) => cleanLine(line));
  const blocks: MessageBlock[] = [];
  let paragraphLines: string[] = [];
  let listItems: string[] = [];

  const flushParagraph = () => {
    if (paragraphLines.length > 0) {
      blocks.push({ type: "paragraph", lines: paragraphLines });
      paragraphLines = [];
    }
  };

  const flushList = () => {
    if (listItems.length > 0) {
      blocks.push({ type: "list", items: listItems });
      listItems = [];
    }
  };

  for (const line of lines) {
    const trimmed = cleanLine(line);

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const listMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (listMatch) {
      flushParagraph();
      listItems.push(listMatch[1]);
      continue;
    }

    const timeLabelMatch = trimmed.match(
      /^(Morning|Afternoon|Evening|Lunch|Dinner|Full Day|Note)\s*:\s*(.*)$/i
    );
    if (timeLabelMatch) {
      flushParagraph();
      listItems.push(
        `${timeLabelMatch[1]}: ${timeLabelMatch[2]}`.replace(/\s+/g, " ").trim()
      );
      continue;
    }

    const markdownHeading = trimmed.match(/^#{1,6}\s+(.+?)\**$/);
    if (markdownHeading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", text: markdownHeading[1].trim() });
      continue;
    }

    const dayHeading = trimmed.match(/^(?:#{1,6}\s*)?(Day\s+\d+[^:]*):?\s*(.*)$/i);
    if (dayHeading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", text: dayHeading[1].trim() });
      if (dayHeading[2]) {
        blocks.push({ type: "paragraph", lines: [dayHeading[2].trim()] });
      }
      continue;
    }

    const sectionHeading = trimmed.match(/^([^:*#]{2,80}):$/);
    if (sectionHeading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", text: sectionHeading[1].trim() });
      continue;
    }

    const placeHeading = trimmed.match(
      /^([A-Za-z][A-Za-z0-9 '&()./-]{4,140},\s*[A-Za-z() .'-]{2,80}(?:,\s*(USA|Mexico|Canada|Australia|New Zealand|Iceland|Norway|Japan|South Africa))?)$/
    );
    if (placeHeading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", text: placeHeading[1].trim() });
      continue;
    }

    const regionHeading = trimmed.match(
      /^([A-Za-z][A-Za-z '&/-]{2,80},\s*(USA|Mexico|Canada|Australia|New Zealand|Iceland|Norway|Japan|South Africa)[^:]{0,80}):$/
    );
    if (regionHeading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "heading", text: regionHeading[1].trim() });
      continue;
    }

    const factMatch = trimmed.match(/^([A-Za-z][^:*#]{2,80}):\s+(.+)$/);
    if (factMatch && !/^Day\s+\d+/i.test(factMatch[1])) {
      flushParagraph();
      flushList();
      blocks.push({
        type: "fact",
        label: factMatch[1].trim(),
        value: factMatch[2].trim(),
      });
      continue;
    }

    flushList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();
  return blocks;
}

function AssistantMessage({ content, blocks }: { content: string; blocks?: StructuredBlock[] }) {
  const safeBlocks = blocks?.length ? blocks : parseMessageBlocks(content);

  return (
    <div className="space-y-3 text-[15px] leading-7 text-slate-700">
      {safeBlocks.map((block, idx) => {
        if (block.type === "heading") {
          return (
            <h4 key={idx} className="text-base font-semibold text-slate-900 pt-1">
              {formatInline(block.text)}
            </h4>
          );
        }

        if (block.type === "fact") {
          return (
            <div
              key={idx}
              className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 leading-6"
            >
              <span className="font-medium text-slate-900">{formatInline(block.label)}:</span>{" "}
              <span className="text-slate-700">{formatInline(block.value)}</span>
            </div>
          );
        }

        if (block.type === "list") {
          return (
            <ul key={idx} className="list-disc pl-5 space-y-2 marker:text-sky-500">
              {block.items.map((item, i) => (
                <li key={i}>{formatInline(item)}</li>
              ))}
            </ul>
          );
        }

        return (
          <p key={idx}>
            {formatInline(block.lines.join(" ").replace(/\s+/g, " "))}
          </p>
        );
      })}
    </div>
  );
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [formData, setFormData] = useState<TripRequest>({
    origin: "Delhi",
    destination: "Jaipur",
    start_date: "2026-04-10",
    end_date: "2026-04-13",
    travelers: 2,
    budget_total: 40000,
    preferences: { type: "cultural", pace: "moderate" },
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TravelPlanResult | null>(null);

  // Chat state
  const [chatMessages, setChatMessages] = useState<UIChatMessage[]>([
    {
      role: "assistant",
      content: WELCOME_MESSAGE,
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatSessionId] = useState(
    () =>
      (typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `session-${Date.now()}`)
  );
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [chatMessages]);

  const handleChatSend = async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    setChatInput("");
    const requestMessages: ChatMessage[] = [
      ...chatMessages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: msg },
    ];
    const assistantIndex = requestMessages.length;
    setChatMessages((prev) => [
      ...prev,
      { role: "user", content: msg },
      { role: "assistant", content: "" },
    ]);
    setChatLoading(true);
    try {
      await streamTravelChat(
        requestMessages,
        (delta) => {
          setChatMessages((prev) => {
            const updated = [...prev];
            const target = updated[assistantIndex];
            if (target && target.role === "assistant") {
              updated[assistantIndex] = {
                ...target,
                content: `${target.content}${delta}`,
              };
            }
            return updated;
          });
        },
        (finalPayload) => {
          setChatMessages((prev) => {
            const updated = [...prev];
            const target = updated[assistantIndex];
            if (target && target.role === "assistant") {
              updated[assistantIndex] = {
                ...target,
                content: finalPayload.message,
                blocks: finalPayload.structured?.blocks,
              };
            }
            return updated;
          });
        },
        chatSessionId
      );
    } catch (err) {
      setChatMessages((prev) => [
        ...prev.slice(0, assistantIndex),
        {
          role: "assistant",
          content:
            err instanceof Error ? err.message : "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleResetChat = () => {
    setChatMessages([{ role: "assistant", content: WELCOME_MESSAGE }]);
    setChatInput("");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await planTrip(formData);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen py-5 px-4 sm:px-6">
      <div className="max-w-7xl mx-auto grid gap-5 lg:grid-cols-[280px_1fr]">
        <aside className="rounded-3xl border border-slate-200/80 bg-white/80 backdrop-blur-xl shadow-sm p-4 sm:p-5 h-fit lg:sticky lg:top-5">
          <div className="flex items-center gap-3 mb-6">
            <div className="h-10 w-10 rounded-2xl bg-sky-600 text-white grid place-content-center text-lg">
              ✈️
            </div>
            <div>
              <h1 className="font-semibold text-slate-900">Travel Copilot</h1>
              <p className="text-xs text-slate-500">Industry-style AI planner</p>
            </div>
          </div>

          <div className="space-y-2 mb-6">
            <button
              type="button"
              onClick={() => setActiveTab("chat")}
              className={`w-full text-left px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                activeTab === "chat"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              💬 AI Chat
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("plan")}
              className={`w-full text-left px-3 py-2.5 rounded-xl text-sm font-medium transition ${
                activeTab === "plan"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              📋 Structured Planner
            </button>
          </div>

        </aside>

        <section className="rounded-3xl border border-slate-200/80 bg-white/80 backdrop-blur-xl shadow-sm overflow-hidden">
          <div className="px-5 sm:px-6 py-4 border-b border-slate-200 flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">
                {activeTab === "chat" ? "Conversational Mode" : "Form Mode"}
              </p>
              <h2 className="text-xl font-semibold text-slate-900">
                {activeTab === "chat"
                  ? "Travel AI Assistant"
                  : "Detailed Trip Planner"}
              </h2>
            </div>
            {activeTab === "chat" && (
              <button
                type="button"
                onClick={handleResetChat}
                className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100"
              >
                New Chat
              </button>
            )}
          </div>

          {/* Chat tab */}
          {activeTab === "chat" && (
            <div className="flex flex-col h-[78vh] min-h-[620px]">
              <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-5 space-y-5">
                {chatMessages.map((m, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-3 ${
                      m.role === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    {m.role === "assistant" && (
                      <div className="h-8 w-8 rounded-full bg-sky-100 text-sky-700 grid place-content-center text-xs font-semibold mt-1">
                        AI
                      </div>
                    )}
                    <div
                      className={`max-w-[88%] sm:max-w-[80%] rounded-2xl px-4 py-3 ${
                        m.role === "user"
                          ? "bg-slate-900 text-white"
                          : "bg-white text-slate-800 border border-slate-200 shadow-sm"
                      }`}
                    >
                      {m.role === "assistant" ? (
                        <AssistantMessage content={m.content} blocks={m.blocks} />
                      ) : (
                        <p className="whitespace-pre-wrap text-sm leading-6">{m.content}</p>
                      )}
                    </div>
                    {m.role === "user" && (
                      <div className="h-8 w-8 rounded-full bg-slate-900/90 text-white grid place-content-center text-xs font-semibold mt-1">
                        You
                      </div>
                    )}
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex items-start gap-3">
                    <div className="h-8 w-8 rounded-full bg-sky-100 text-sky-700 grid place-content-center text-xs font-semibold mt-1">
                      AI
                    </div>
                    <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-sm">
                      <div className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full bg-slate-300 animate-pulse" />
                        <span className="h-2 w-2 rounded-full bg-slate-300 animate-pulse [animation-delay:120ms]" />
                        <span className="h-2 w-2 rounded-full bg-slate-300 animate-pulse [animation-delay:240ms]" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <div className="px-4 sm:px-6 py-4 border-t border-slate-200 bg-white/70">
                <div className="rounded-2xl border border-slate-300 bg-white p-2.5 flex items-end gap-2">
                  <textarea
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        void handleChatSend();
                      }
                    }}
                    placeholder="Ask for itinerary, budget split, flights + stay suggestions..."
                    rows={1}
                    className="flex-1 resize-none outline-none text-sm leading-6 px-2 py-1 max-h-36"
                  />
                  <button
                    type="button"
                    onClick={handleChatSend}
                    disabled={chatLoading || !chatInput.trim()}
                    className="h-10 px-4 rounded-xl bg-sky-600 hover:bg-sky-700 disabled:bg-slate-300 disabled:text-slate-500 text-white text-sm font-medium transition"
                  >
                    Send
                  </button>
                </div>
                <p className="text-xs text-slate-500 mt-2 px-1">
                  Press Enter to send, Shift+Enter for new line.
                </p>
              </div>
            </div>
          )}

      {/* Form tab */}
      {activeTab === "plan" && (
      <>
      {/* Form */}
      <form
        onSubmit={handleSubmit}
        className="bg-white rounded-2xl shadow-lg p-6 mb-10 border border-slate-200"
      >
        <h2 className="text-xl font-semibold text-slate-800 mb-4">
          ✈️ Tell Us Your Trip Details
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Where are you departing from
            </label>
            <input
              type="text"
              value={formData.origin}
              onChange={(e) =>
                setFormData({ ...formData, origin: e.target.value })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              placeholder="e.g. Delhi"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Where are you going
            </label>
            <input
              type="text"
              value={formData.destination}
              onChange={(e) =>
                setFormData({ ...formData, destination: e.target.value })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              placeholder="e.g. Jaipur"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Start Date
            </label>
            <input
              type="date"
              value={formData.start_date}
              onChange={(e) =>
                setFormData({ ...formData, start_date: e.target.value })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              End Date
            </label>
            <input
              type="date"
              value={formData.end_date}
              onChange={(e) =>
                setFormData({ ...formData, end_date: e.target.value })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Number of travelers
            </label>
            <input
              type="number"
              min={1}
              max={20}
              value={formData.travelers || 1}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  travelers: parseInt(e.target.value) || 1,
                })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Budget (₹)
            </label>
            <input
              type="number"
              min={1000}
              value={formData.budget_total || 50000}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  budget_total: parseInt(e.target.value) || 50000,
                })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Trip Type
            </label>
            <select
              value={formData.preferences?.type || "leisure"}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  preferences: {
                    ...formData.preferences!,
                    type: e.target.value,
                  },
                })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            >
              {PREFERENCE_OPTIONS.type.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">
              Pace
            </label>
            <select
              value={formData.preferences?.pace || "relaxed"}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  preferences: {
                    ...formData.preferences!,
                    pace: e.target.value,
                  },
                })
              }
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            >
              {PREFERENCE_OPTIONS.pace.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 px-6 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-400 text-white font-semibold rounded-xl transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="animate-spin h-5 w-5"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Creating your plan... Please wait!
            </span>
          ) : (
            "🚀 Create Plan!"
          )}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
          ⚠️ {error}
        </div>
      )}

      {/* Results */}
      {result && <ResultsView data={result} />}
      </>
      )}
        </section>
      </div>
    </main>
  );
}

function ResultsView({ data }: { data: TravelPlanResult }) {
  const bb = data.budget_breakdown || {};
  const skipKeys = [
    "total_estimated",
    "per_person",
    "surplus_or_deficit",
    "budget_available",
  ];

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold text-slate-800 text-center">
        🎊 Your Travel Plan is Ready!
      </h2>

      {/* Summary */}
      <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
        <h3 className="text-lg font-semibold text-sky-700 mb-4">
          📋 Trip Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-slate-500">From</span>
            <p className="font-medium">{data.origin}</p>
          </div>
          <div>
            <span className="text-slate-500">To</span>
            <p className="font-medium">{data.destination}</p>
          </div>
          <div>
            <span className="text-slate-500">Duration</span>
            <p className="font-medium">
              {data.start_date} → {data.end_date}
            </p>
          </div>
          <div>
            <span className="text-slate-500">Travelers</span>
            <p className="font-medium">{data.travelers} people</p>
          </div>
        </div>
      </section>

      {/* Budget Breakdown */}
      <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
        <h3 className="text-lg font-semibold text-sky-700 mb-4">
          💰 Budget Breakdown
        </h3>
        <div className="space-y-2 text-sm">
          {Object.entries(bb).map(
            ([key, val]) =>
              !skipKeys.includes(key) &&
              typeof val === "number" && (
                <div
                  key={key}
                  className="flex justify-between py-1 border-b border-slate-100"
                >
                  <span className="text-slate-600 capitalize">
                    {key.replace(/_/g, " ")}
                  </span>
                  <span className="font-medium">₹{val.toLocaleString("en-IN")}</span>
                </div>
              )
          )}
          <div className="flex justify-between pt-3 font-semibold text-base">
            <span>Total</span>
            <span>₹{(bb.total_estimated || 0).toLocaleString("en-IN")}</span>
          </div>
          <div className="flex justify-between text-slate-600">
            <span>Per Person</span>
            <span>₹{(bb.per_person || 0).toLocaleString("en-IN")}</span>
          </div>
          {(bb.surplus_or_deficit ?? 0) >= 0 ? (
            <p className="text-green-600 font-medium pt-2">
              ✅ Within budget! You&apos;ll save ₹{(bb.surplus_or_deficit || 0).toLocaleString("en-IN")}
            </p>
          ) : (
            <p className="text-amber-600 font-medium pt-2">
              ⚠️ ₹{Math.abs(bb.surplus_or_deficit || 0).toLocaleString("en-IN")}{" "}
              extra required
            </p>
          )}
        </div>
      </section>

      {/* Day-wise Itinerary */}
      {data.itinerary && data.itinerary.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            📅 Day-wise Itinerary
          </h3>
          <div className="space-y-6">
            {data.itinerary.map((day) => (
              <div
                key={day.day}
                className="border-l-4 border-sky-400 pl-4 py-2"
              >
                <h4 className="font-semibold text-slate-800">
                  Day {day.day}: {day.title}
                </h4>
                <ul className="mt-2 space-y-1 text-sm text-slate-600">
                  {day.schedule?.slice(0, 6).map((item, i) => (
                    <li key={i}>
                      <span className="font-medium text-slate-500">
                        {item.time}
                      </span>{" "}
                      {item.activity}
                    </li>
                  ))}
                  {(day.schedule?.length || 0) > 6 && (
                    <li className="text-slate-500 italic">
                      + {day.schedule!.length - 6} more activities
                    </li>
                  )}
                </ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Sightseeing */}
      {data.sightseeing && data.sightseeing.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            🗺️ Places to Visit
          </h3>
          <div className="grid gap-4 sm:grid-cols-2">
            {data.sightseeing.slice(0, 6).map((place, i) => (
              <div
                key={i}
                className="p-4 bg-slate-50 rounded-xl border border-slate-100"
              >
                <h4 className="font-medium text-slate-800">{place.name}</h4>
                {place.description && (
                  <p className="text-sm text-slate-600 mt-1">
                    {place.description}
                  </p>
                )}
                <div className="mt-2 flex gap-3 text-xs text-slate-500">
                  {place.entry_fee != null && (
                    <span>Entry: ₹{place.entry_fee}</span>
                  )}
                  {place.time_needed && <span>Time: {place.time_needed}</span>}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Restaurants */}
      {data.restaurants && data.restaurants.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            🍽️ Restaurants
          </h3>
          <div className="space-y-3">
            {data.restaurants.slice(0, 5).map((r, i) => (
              <div
                key={i}
                className="flex justify-between items-start p-3 bg-amber-50 rounded-lg border border-amber-100"
              >
                <div>
                  <h4 className="font-medium text-slate-800">
                    {r.name}
                    {r.must_try && (
                      <span className="ml-2 text-amber-600 text-xs">Must try!</span>
                    )}
                  </h4>
                  {r.specialty && (
                    <p className="text-sm text-slate-600">{r.specialty}</p>
                  )}
                </div>
                <div className="text-right text-sm">
                  {r.avg_cost_per_person != null && (
                    <p>₹{r.avg_cost_per_person}/person</p>
                  )}
                  {r.rating != null && (
                    <p className="text-amber-600">{r.rating}⭐</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Tips */}
      {data.tips && data.tips.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            💡 Smart Tips
          </h3>
          <ul className="space-y-2 text-sm text-slate-700">
            {data.tips.slice(0, 10).map((tip, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-amber-500">•</span> {tip}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Packing List */}
      {data.packing_list && data.packing_list.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            🎒 Packing List
          </h3>
          <div className="flex flex-wrap gap-2">
            {data.packing_list.slice(0, 16).map((item, i) => (
              <span
                key={i}
                className="px-3 py-1 bg-slate-100 rounded-full text-sm text-slate-700"
              >
                {item}
              </span>
            ))}
            {data.packing_list.length > 16 && (
              <span className="text-slate-500 text-sm">
                +{data.packing_list.length - 16} more
              </span>
            )}
          </div>
        </section>
      )}

      {/* Recommendations */}
      {data.recommendations && data.recommendations.length > 0 && (
        <section className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
          <h3 className="text-lg font-semibold text-sky-700 mb-4">
            ✅ Final Recommendations
          </h3>
          <ul className="space-y-2 text-sm text-slate-700">
            {data.recommendations.map((rec, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-green-500">✓</span> {rec}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
