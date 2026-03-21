"use client";

import { useEffect, useRef, useState } from "react";

import { streamTravelChat } from "@/lib/api";
import type { ChatMessage, StructuredBlock } from "@/types";

const WELCOME_MESSAGE =
  "Hi! I'm your travel assistant. Tell me where you'd like to go, your budget, dates, or what kind of trip you're looking for - I'll help you plan it!";

type UIChatMessage = ChatMessage & { blocks?: StructuredBlock[] };

function renderInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={idx} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

function AssistantMessage({ message }: { message: UIChatMessage }) {
  const blocks = message.blocks;
  if (!blocks || blocks.length === 0) {
    return <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>;
  }

  return (
    <div className="space-y-3 text-sm leading-6">
      {blocks.map((block, i) => {
        if (block.type === "heading") {
          return (
            <h4 key={i} className="text-base font-semibold text-slate-900">
              {renderInline(block.text)}
            </h4>
          );
        }
        if (block.type === "fact") {
          return (
            <p key={i}>
              <span className="font-semibold">{renderInline(block.label)}:</span> {renderInline(block.value)}
            </p>
          );
        }
        if (block.type === "list") {
          return (
            <ul key={i} className="list-disc pl-5 space-y-1">
              {block.items.map((item, idx) => (
                <li key={idx}>{renderInline(item)}</li>
              ))}
            </ul>
          );
        }
        return (
          <p key={i} className="whitespace-pre-wrap">
            {renderInline(block.lines.join(" "))}
          </p>
        );
      })}
    </div>
  );
}

export default function Home() {
  const createSessionId = () =>
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `session-${Date.now()}`;

  const [chatMessages, setChatMessages] = useState<UIChatMessage[]>([
    { role: "assistant", content: WELCOME_MESSAGE },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId, setSessionId] = useState(() => createSessionId());

  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, chatLoading]);

  const handleChatSend = async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;

    setChatInput("");
    const requestMessages: ChatMessage[] = [
      ...chatMessages.map((m) => ({ role: m.role, content: m.content })),
      { role: "user", content: msg },
    ];
    const assistantIndex = requestMessages.length;

    setChatMessages((prev) => [...prev, { role: "user", content: msg }, { role: "assistant", content: "" }]);
    setChatLoading(true);

    try {
      await streamTravelChat(
        requestMessages,
        (delta) => {
          setChatMessages((prev) => {
            const next = [...prev];
            const target = next[assistantIndex];
            if (target && target.role === "assistant") {
              next[assistantIndex] = { ...target, content: `${target.content}${delta}` };
            }
            return next;
          });
        },
        (finalPayload) => {
          setChatMessages((prev) => {
            const next = [...prev];
            const target = next[assistantIndex];
            if (target && target.role === "assistant") {
              next[assistantIndex] = {
                ...target,
                content: finalPayload.message,
                blocks: finalPayload.structured?.blocks,
              };
            }
            return next;
          });
        },
        sessionId
      );
    } catch (error) {
      setChatMessages((prev) => [
        ...prev.slice(0, assistantIndex),
        {
          role: "assistant",
          content: error instanceof Error ? error.message : "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <main className="min-h-screen py-6 px-4 sm:px-6">
      <div className="max-w-5xl mx-auto rounded-3xl border border-slate-200 bg-white shadow-sm overflow-hidden">
        <header className="px-5 sm:px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-500">Travel Chatbot</p>
            <h1 className="text-xl font-semibold text-slate-900">Travel AI Assistant</h1>
          </div>
          <button
            type="button"
            onClick={() => {
              setChatMessages([{ role: "assistant", content: WELCOME_MESSAGE }]);
              setChatInput("");
              setSessionId(createSessionId());
            }}
            className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 text-slate-700 hover:bg-slate-100"
          >
            New Chat
          </button>
        </header>

        <section className="flex flex-col h-[78vh] min-h-[620px]">
          <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-5 space-y-4">
            {chatMessages.map((m, i) => (
              <div key={i} className={`flex items-start gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                {m.role === "assistant" && (
                  <div className="h-8 w-8 rounded-full bg-sky-100 text-sky-700 grid place-content-center text-xs font-semibold mt-1">
                    AI
                  </div>
                )}
                <div
                  className={`max-w-[90%] sm:max-w-[80%] rounded-2xl px-4 py-3 ${
                    m.role === "user" ? "bg-slate-900 text-white" : "bg-white text-slate-800 border border-slate-200"
                  }`}
                >
                  {m.role === "assistant" ? (
                    <AssistantMessage message={m} />
                  ) : (
                    <p className="whitespace-pre-wrap text-sm leading-6">{m.content}</p>
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          <footer className="px-4 sm:px-6 py-4 border-t border-slate-200 bg-white">
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
          </footer>
        </section>
      </div>
    </main>
  );
}
