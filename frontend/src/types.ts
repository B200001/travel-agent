// API Request
export interface TripPreferences {
  type: string;   // leisure, cultural, adventure, party, family
  pace: string;   // relaxed, moderate, packed
}

export interface TripRequest {
  origin: string;
  destination: string;
  start_date: string;
  end_date: string;
  travelers?: number;
  budget_total?: number;
  preferences?: TripPreferences;
}

// API Response - Travel Plan Result
export interface ItineraryItem {
  time: string;
  activity: string;
}

export interface DayPlan {
  day: number;
  title: string;
  schedule: ItineraryItem[];
}

export interface SightseeingPlace {
  priority?: number;
  name: string;
  description?: string;
  entry_fee?: number;
  time_needed?: string;
}

export interface Restaurant {
  name: string;
  specialty?: string;
  avg_cost_per_person?: number;
  rating?: number;
  must_try?: boolean;
}

export interface Hotel {
  name?: string;
  [key: string]: unknown;
}

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

export interface TravelPlanResult {
  origin: string;
  destination: string;
  start_date: string;
  end_date: string;
  travelers: number;
  budget_total: number;
  flights: Record<string, unknown>;
  hotels: Hotel[];
  local_transport: Record<string, unknown>;
  restaurants: Restaurant[];
  sightseeing: SightseeingPlace[];
  activities: unknown[];
  weather_forecast: Record<string, unknown>;
  packing_list: string[];
  itinerary: DayPlan[];
  budget_breakdown: {
    total_estimated?: number;
    per_person?: number;
    surplus_or_deficit?: number;
    budget_available?: number;
    [key: string]: number | undefined;
  };
  recommendations: string[];
  tips: string[];
  warnings: string[];
}
