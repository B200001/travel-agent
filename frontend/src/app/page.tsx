"use client";

import { useState, useRef, useEffect } from "react";
import { planTrip, sendTravelChat } from "@/lib/api";
import type { TripRequest, TravelPlanResult, ChatMessage } from "@/types";

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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I'm your travel assistant. Tell me where you'd like to go, your budget, dates, or what kind of trip you're looking for - I'll help you plan it!",
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [chatMessages]);

  const handleChatSend = async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", content: msg }]);
    setChatLoading(true);
    try {
      const nextMessages: ChatMessage[] = [
        ...chatMessages,
        { role: "user", content: msg },
      ];
      const res = await sendTravelChat(nextMessages);
      setChatMessages((prev) => [...prev, { role: "assistant", content: res.message }]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: err instanceof Error ? err.message : "Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setChatLoading(false);
    }
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
    <main className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto">
      {/* Header */}
      <header className="text-center mb-10">
        <h1 className="text-4xl font-bold text-sky-800 mb-2">
          🌟 Travel Agent
        </h1>
        <p className="text-slate-600 text-lg">
          Flights, hotels, restaurants, sightseeing - we&apos;ll plan it all!
        </p>
      </header>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b border-slate-200">
        <button
          type="button"
          onClick={() => setActiveTab("chat")}
          className={`px-4 py-2 font-medium rounded-t-lg transition-colors ${
            activeTab === "chat"
              ? "bg-sky-100 text-sky-700 border-b-2 border-sky-600 -mb-px"
              : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          💬 Chat
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("plan")}
          className={`px-4 py-2 font-medium rounded-t-lg transition-colors ${
            activeTab === "plan"
              ? "bg-sky-100 text-sky-700 border-b-2 border-sky-600 -mb-px"
              : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          📋 Plan Trip (Form)
        </button>
      </div>

      {/* Chat tab */}
      {activeTab === "chat" && (
        <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden flex flex-col h-[600px]">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {chatMessages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2 ${
                    m.role === "user"
                      ? "bg-sky-600 text-white"
                      : "bg-slate-100 text-slate-800"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm">{m.content}</p>
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-slate-100 rounded-2xl px-4 py-2 text-slate-500 text-sm">
                  Typing...
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div className="p-4 border-t border-slate-200 flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleChatSend()}
              placeholder="Ask about destinations, budgets, itineraries..."
              className="flex-1 px-4 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-sky-500"
            />
            <button
              type="button"
              onClick={handleChatSend}
              disabled={chatLoading}
              className="px-5 py-2 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-400 text-white font-medium rounded-xl"
            >
              Send
            </button>
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
