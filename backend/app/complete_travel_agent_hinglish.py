from typing import TypedDict, Annotated, List, Dict, Optional
from datetime import datetime, timedelta
import operator
import os
import re
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from google import genai

load_dotenv()


class CompleteTravelState(TypedDict):
    """The state of the complete travel agent. And going to maintain
    information storage for the travel agent.
    """

    # basic details of user
    origin: str
    destination: str
    start_date: str  # "YYYY-MM-DD" format
    end_date: str    # "YYYY-MM-DD" format
    travelers: int
    budget_total: float
    preferences: Dict

    # Travel components to book
    flights: Dict
    hotels: List[Dict]
    local_transport: Dict
    restaurants: List[Dict]
    sightseeing: List[Dict]
    activities: List[Dict]
    total_cost: float
    total_duration: timedelta
    total_distance: float
    total_activities: int
    total_restaurants: int

    # planning data
    weather_forecast: Dict
    packing_list: List[str]
    itinerary: List[Dict]
    budget_breakdown: Dict

    # intelligent suggestions
    recommendations: List[Dict]
    tips: List[Dict]
    warnings: List[Dict]

    # Conversation
    messages: Annotated[list, operator.add]
    current_step: str



class CompleteTravelAgent:
    """
    the main agent which will manage everything like a travel agent in an office.
    """

    def __init__(self, api_key: str):
        """
        Initialize the agent with a Gemini API key (GEMINI_API_KEY or GOOGLE_API_KEY).
        """
        self.client = genai.Client(api_key=api_key)
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.model = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"
        self.graph = self._build_complete_graph()

    def _parse_json_lenient(self, content: str):
        """Parse JSON from LLM output. Handles markdown blocks, trailing commas, and extracts first {...} or [...]."""
        content = content.strip()
        # Strip markdown code block if present
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if match:
            content = match.group(1).strip()
        # Find first JSON-like start
        start_obj = content.find('{')
        start_arr = content.find('[')
        if start_obj == -1 and start_arr == -1:
            raise ValueError("No JSON object or array found")
        start = min((start_obj if start_obj >= 0 else 1 << 30), (start_arr if start_arr >= 0 else 1 << 30))
        depth = 0
        in_string = False
        escape = False
        quote = None
        end = -1
        for i in range(start, len(content)):
            c = content[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if in_string:
                if c == quote:
                    in_string = False
                continue
            if c in '"\'':
                in_string = True
                quote = c
                continue
            if c in '{[':
                depth += 1
            elif c in '}]':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            raise ValueError("Unclosed JSON")
        raw = content[start:end]
        # Fix trailing commas (invalid in strict JSON)
        raw = re.sub(r',\s*}', '}', raw)
        raw = re.sub(r',\s*]', ']', raw)
        return json.loads(raw)

    def _call_llm_json(self, prompt: str, json_format: str, fallback) -> any:
        """
        Call the LLM and return parsed JSON. Uses lenient parsing so that
        real destination-specific data is used even when the model wraps JSON
        in markdown or adds trailing commas.
        """
        full_prompt = f"""{prompt}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanation - raw JSON only.
Expected structure: {json_format}
"""
        content = ""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )
            content = (response.text or "").strip()
            if not content:
                raise ValueError("Empty LLM response")
            return self._parse_json_lenient(content)
        except Exception as e:
            snippet = (content[:400] + "...") if len(content) > 400 else content or "(no content)"
            print(f"[LLM fallback used] {e}\nResponse snippet: {snippet}")
            return fallback
        
    
    def _build_complete_graph(self) -> StateGraph:
        """
        define the workflow like which step to take next.

        flow:
        1. search for flights
        2. search for hotels
        3. plan for local transport
        4. suggestions for restaurants
        5. plan for sightseeing
        5a. suggestions for activities
        6. checking weather
        7. packing list creation
        8. ready complete itinerary
        9. calculation for total budget
        10. final recommendations
        """

        workflow = StateGraph(CompleteTravelState)

        # define nodes for the workflow

        workflow.add_node("search_flights", self.search_flights)
        workflow.add_node("find_hotels", self.find_hotels)
        workflow.add_node("plan_transport", self.plan_local_transport)
        workflow.add_node("suggest_restaurants", self.suggest_restaurants)
        workflow.add_node("plan_sightseeing", self.plan_sightseeing)
        workflow.add_node("find_activities", self.find_activities)
        workflow.add_node("check_weather", self.check_weather)
        workflow.add_node("create_packing_list", self.create_packing_list)
        workflow.add_node("build_itinerary", self.build_day_itinerary)
        workflow.add_node("calculate_budget", self.calculate_budget)
        workflow.add_node("generate_tips", self.generate_smart_tips)
        workflow.add_node("finalize_plan", self.finalize_complete_plan)

        # define edges for the workflow

        workflow.set_entry_point("search_flights")
        workflow.add_edge("search_flights", "find_hotels")
        workflow.add_edge("find_hotels", "plan_transport")
        workflow.add_edge("plan_transport", "suggest_restaurants")
        workflow.add_edge("suggest_restaurants", "plan_sightseeing")
        workflow.add_edge("plan_sightseeing", "find_activities")
        workflow.add_edge("find_activities", "check_weather")
        workflow.add_edge("check_weather", "create_packing_list")
        workflow.add_edge("create_packing_list", "build_itinerary")
        workflow.add_edge("build_itinerary", "calculate_budget")
        workflow.add_edge("calculate_budget", "generate_tips")
        workflow.add_edge("generate_tips", "finalize_plan")
        workflow.add_edge("finalize_plan", END)
        return workflow.compile()

    def search_flights(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        search for flights and finds the best options
        - will check for multiple airlines
        -compare the price
        - check on time performance
        - recommends the best option
        """

        print("Searching for flights...")

        origin, dest = state["origin"], state["destination"]
        prompt = f"""
        Suggest realistic flight options for this trip:
        From: {origin}
        To: {dest}
        Date: {state['start_date']} (outbound), Return: {state['end_date']}
        Travelers: {state['travelers']}, Budget: ₹{state['budget_total']}
        
        Use airlines that actually fly this route (e.g. for international from India: Air India, IndiGo, Vistara, or relevant international carriers). Return realistic departure/arrival times and prices in INR.
        Return JSON: onward (airline, flight_number, departure, arrival, price, direct), return (same fields), total_cost (number).
        """
        fallback_flights = {
            "onward": {"airline": "Flight", "flight_number": "XX-001", "departure": "14:30",
                       "arrival": "17:45", "price": 15000, "direct": True},
            "return": {"airline": "Flight", "flight_number": "XX-002", "departure": "19:00",
                       "arrival": "22:15", "price": 15000, "direct": True},
            "total_cost": 30000
        }
        flights = self._call_llm_json(prompt, '{"onward": {...}, "return": {...}, "total_cost": number}', fallback_flights)
        state["flights"] = flights
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "✅ Best flights mil gayi! IndiGo recommended - on-time aur affordable"}
        ]
        
        print("✅ Flights ready!")
        return state

    def find_hotels(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        search for hotels based on budget and location
        - ratings with amenities
        """
        print("Searching for hotels...")

        # calculate nights
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        nights = (end - start).days

        hotel_budget_per_night = (state["budget_total"] * 0.35) / nights

        prompt = f"""
        Find hotels in {state['destination']} for {nights} nights.
        Budget per night: ₹{hotel_budget_per_night}
        Travelers: {state['travelers']}
        
        Consider:
        1. Central location (main attractions ke paas)
        2. Good ratings (4+ stars on Google)
        3. Amenities (WiFi, breakfast)
        4. Value for money
        
        Suggest 3 options: Budget, Mid-range, Luxury. Return array of hotel objects.
        """
        
        fallback_hotels = [
            {"name": "Hotel Comfort Inn", "type": "budget", "location": "City Center",
             "price_per_night": hotel_budget_per_night * 0.7, "total_cost": hotel_budget_per_night * 0.7 * nights,
             "rating": 4.1, "amenities": ["WiFi", "Breakfast", "AC"], "distance_from_center": "2 km"},
            {"name": "Grand Stay Hotel", "type": "mid-range", "location": "Main Market",
             "price_per_night": hotel_budget_per_night, "total_cost": hotel_budget_per_night * nights,
             "rating": 4.4, "amenities": ["WiFi", "Breakfast", "Pool", "Gym"], "distance_from_center": "500m"},
            {"name": "Luxury Palace Hotel", "type": "luxury", "location": "Premium Area",
             "price_per_night": hotel_budget_per_night * 1.5, "total_cost": hotel_budget_per_night * 1.5 * nights,
             "rating": 4.7, "amenities": ["WiFi", "Breakfast", "Pool", "Spa", "Restaurant"], "distance_from_center": "1 km"}
        ]
        hotels = self._call_llm_json(prompt, '[{"name","type","location","price_per_night","total_cost","rating","amenities","distance_from_center"}]', fallback_hotels)
        state["hotels"] = hotels if isinstance(hotels, list) else fallback_hotels
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"✅ {len(state['hotels'])} hotel options mil gaye! Budget se luxury tak"}
        ]
        
        print("✅ Hotels ready!")
        return state

    def plan_local_transport(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        plans local transport

        options:
        -cabs, metro, buses, etc.
        """
        print("Planning local transport...")

        prompt = f"""
        Plan local transport for {state['destination']}.
        Duration: {(datetime.strptime(state['end_date'], '%Y-%m-%d') - datetime.strptime(state['start_date'], '%Y-%m-%d')).days} days
        Travelers: {state['travelers']}
        
        Consider:
        1. Most cost-effective option
        2. Convenience
        3. Safety
        4. Availability
        
        Suggest best transport mix. Return primary, secondary, backup (each with type, estimated_daily_cost, pros, cons), and recommended string.
        """
        
        fallback_transport = {
            "primary": {"type": "Ola/Uber", "estimated_daily_cost": 800, "pros": ["Door to door", "AC comfort", "Safe"], "cons": ["Surge pricing possible"]},
            "secondary": {"type": "Metro/Bus", "estimated_daily_cost": 150, "pros": ["Very cheap", "No traffic", "Eco-friendly"], "cons": ["Fixed routes", "Crowded"]},
            "backup": {"type": "Auto-rickshaw", "estimated_daily_cost": 400, "pros": ["Easy to find", "Quick for short distances"], "cons": ["No AC", "Bargaining needed"]},
            "recommended": "Mix of Metro for main routes + Uber for convenience"
        }
        transport = self._call_llm_json(prompt, '{"primary":{...},"secondary":{...},"backup":{...},"recommended":"string"}', fallback_transport)
        state["local_transport"] = transport if isinstance(transport, dict) and "primary" in transport else fallback_transport
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "✅ Transport plan ready! Metro + Uber combination best rahega"}
        ]
        
        print("✅ Transport planned!")
        return state
    
    def suggest_restaurants(self, state:CompleteTravelState) -> CompleteTravelState:
        """
         suggest best restaurants
        """
        print("Suggesting restaurants...")
        dest = state["destination"]
        prompt = f"""
        Suggest best restaurants and food spots IN {dest} only. Use REAL or realistic {dest}-specific names and local cuisine (e.g. for Japan: ramen, sushi, izakaya; for Paris: bistros, patisseries). Do NOT use generic Indian names like dhaba, paneer, or Indian dishes unless the destination is in India.
        
        Destination: {dest}
        Travelers: {state['travelers']}
        
        Categories: 1) Must-try local cuisine 2) Budget-friendly 3) Popular/famous 4) Street food 5) Fine dining (if budget allows).
        Return array of 4-5 restaurant objects with name, type, cuisine, avg_cost_per_person (number), rating, specialty, must_try (boolean). Names and cuisine must be specific to {dest}.
        """
        
        fallback_restaurants = [
            {"name": "Local Favourite", "type": "authentic_local", "cuisine": "Local", "avg_cost_per_person": 300, "rating": 4.5, "specialty": "Local specialties", "must_try": True},
            {"name": "Street Food Spot", "type": "street_food", "cuisine": "Street food", "avg_cost_per_person": 150, "rating": 4.3, "specialty": "Popular snacks", "must_try": True},
            {"name": "Mid-Range Restaurant", "type": "casual_dining", "cuisine": "Mixed", "avg_cost_per_person": 500, "rating": 4.2, "specialty": "Various options", "must_try": False},
            {"name": "Fine Dining Option", "type": "fine_dining", "cuisine": "Contemporary", "avg_cost_per_person": 1200, "rating": 4.6, "specialty": "Special occasion", "must_try": False}
        ]
        restaurants = self._call_llm_json(prompt, '[{"name","type","cuisine","avg_cost_per_person","rating","specialty","must_try"}]', fallback_restaurants)
        state["restaurants"] = restaurants if isinstance(restaurants, list) else fallback_restaurants
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "✅ Best food spots mil gaye! Local se fine dining tak"}
        ]
        
        print("✅ Restaurants ready!")
        return state

    def plan_sightseeing(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        suggest best sightseeing spots based on priority
        - must-visit attractions
        """
        print("Planning sightseeing...")

        dest = state["destination"]
        num_days = (datetime.strptime(state["end_date"], "%Y-%m-%d") - datetime.strptime(state["start_date"], "%Y-%m-%d")).days
        prompt = f"""
        Plan sightseeing IN {dest} only. Use REAL or well-known attraction names for {dest} (e.g. for Japan: Senso-ji, Fushimi Inari, Tokyo Skytree; for Paris: Eiffel Tower, Louvre). Do NOT use generic placeholders like "Famous Monument" or "Historical Fort".
        
        Destination: {dest}
        Days available: {num_days}
        
        Categorize: 1) Must-visit (top 3-5) 2) Good to visit 3) Hidden gems.
        Each object: name (destination-specific), category, priority (number), entry_fee (number), time_needed (e.g. "2-3 hours"), best_time, rating, description.
        Return array of 5-6 sightseeing spots with real/relevant names for {dest}.
        """
        
        fallback_sightseeing = [
            {"name": "Top Landmark", "category": "must_visit", "priority": 1, "entry_fee": 500, "time_needed": "2-3 hours", "best_time": "Morning 8-11 AM", "rating": 4.7, "description": "Iconic attraction"},
            {"name": "Historic Site", "category": "must_visit", "priority": 2, "entry_fee": 300, "time_needed": "2 hours", "best_time": "Afternoon 3-5 PM", "rating": 4.5, "description": "Cultural heritage"},
            {"name": "Local Market", "category": "must_visit", "priority": 3, "entry_fee": 0, "time_needed": "1-2 hours", "best_time": "Evening 6-8 PM", "rating": 4.4, "description": "Shopping and local food"},
            {"name": "Museum", "category": "good_to_visit", "priority": 4, "entry_fee": 200, "time_needed": "1.5 hours", "best_time": "Morning 10-12 PM", "rating": 4.2, "description": "Educational"},
            {"name": "Hidden Gem", "category": "hidden_gem", "priority": 5, "entry_fee": 0, "time_needed": "1 hour", "best_time": "Early morning 6-8 AM", "rating": 4.6, "description": "Off-the-beaten-path"}
        ]
        sightseeing = self._call_llm_json(prompt, '[{"name","category","priority","entry_fee","time_needed","best_time","rating","description"}]', fallback_sightseeing)
        state["sightseeing"] = sightseeing if isinstance(sightseeing, list) else fallback_sightseeing
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"✅ {len(state['sightseeing'])} amazing places mil gaye!"}
        ]
        
        print("✅ Sightseeing planned!")
        return state
    
    def find_activities(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        suggest best activities based on interest
        - outdoor, indoor, cultural, etc.
        """
        print("Finding activities...")

        dest = state["destination"]
        prompt = f"""
        Suggest 3-4 fun activities IN {dest} only. Use destination-specific activity names (e.g. for Japan: tea ceremony, sumo show, manga museum; for Thailand: cooking class, temple tour). Do NOT use generic names like "City Walking Tour" only—make them specific to {dest}.
        Destination: {dest}, Travelers: {state['travelers']}, Budget: ₹{state['budget_total']}
        Return array with name, type, duration, cost (number), rating, description.
        """
        fallback_activities = [
            {"name": "Local Walking Tour", "type": "cultural", "duration": "3 hours", "cost": 800, "rating": 4.6, "description": "Guided city explore"},
            {"name": "Food Experience", "type": "culinary", "duration": "2 hours", "cost": 1200, "rating": 4.7, "description": "Local food tour"},
            {"name": "Adventure Activity", "type": "adventure", "duration": "4 hours", "cost": 2500, "rating": 4.5, "description": "Outdoor experience"}
        ]
        activities = self._call_llm_json(prompt, '[{"name","type","duration","cost","rating","description"}]', fallback_activities)
        state["activities"] = activities if isinstance(activities, list) else fallback_activities
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "✅ Fun activities mil gayi! Optional but recommended"}
        ]
        
        print("✅ Activities ready!")
        return state

    def check_weather(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        check weather for the destination
        - temperature, humidity, precipitation
        """
        print("Checking weather...")

        prompt = f"""
        For {state['destination']}, dates {state['start_date']} to {state['end_date']}, predict typical weather.
        Return JSON: general (string), temperature (day, night strings), rainfall, humidity, best_days (array), precautions (array).
        """
        fallback_weather = {
            "general": "Mostly sunny with occasional clouds",
            "temperature": {"day": "28-32°C", "night": "18-22°C"},
            "rainfall": "10% chance", "humidity": "65%",
            "best_days": ["Day 2", "Day 3"],
            "precautions": ["Carry sunscreen", "Umbrella recommended", "Light cotton clothes best"]
        }
        weather = self._call_llm_json(prompt, '{"general","temperature":{...},"rainfall","humidity","best_days","precautions"}', fallback_weather)
        state["weather_forecast"] = weather if isinstance(weather, dict) and "general" in weather else fallback_weather
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "✅ Weather achha rahega! Mostly sunny"}
        ]
        
        print("✅ Weather checked!")
        return state

    def create_packing_list(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        create a packing list based on weather and activities
        - clothing, accessories, toiletries
        """
        print("Creating packing list...")
        
        prompt = f"""
        Create packing list for:
        Destination: {state['destination']}
        Duration: {(datetime.strptime(state['end_date'], '%Y-%m-%d') - datetime.strptime(state['start_date'], '%Y-%m-%d')).days} days
        Weather: {state.get('weather_forecast', {}).get('general', 'Moderate')}
        Activities: Sightseeing, Restaurants
        
        Return JSON array of strings - each item one packing item (e.g. "📄 ID Proof", "👕 3-4 T-shirts").
        Include: documents, clothes (weather-appropriate), electronics, toiletries, misc.
        """
        
        fallback_packing = [
            "📄 ID Proof (Aadhar/Passport)", "📄 Flight tickets", "💳 Cards", "💵 Cash",
            "👕 3-4 T-shirts", "👖 2-3 Pants/Jeans", "👟 Walking shoes", "🩴 Slippers",
            "📱 Phone + Charger", "🔌 Power bank", "🪥 Toothbrush, Toothpaste",
            "🧴 Sunscreen", "🕶️ Sunglasses", "🎒 Day backpack", "💧 Water bottle", "🌂 Umbrella"
        ]
        packing = self._call_llm_json(prompt, '["item1", "item2", ...]', fallback_packing)
        state["packing_list"] = packing if isinstance(packing, list) and all(isinstance(x, str) for x in packing) else fallback_packing
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"✅ Complete packing list ready! {len(state['packing_list'])} items"}
        ]
        
        print("✅ Packing list ready!")
        return state
    
    def build_day_itinerary(self, state: CompleteTravelState) -> CompleteTravelState:
        """
        Build day itinerary from state: uses actual restaurants, sightseeing,
        activities, and flight times so every trip is destination-specific.
        """
        print("Building day itinerary...")

        days_total = (datetime.strptime(state["end_date"], "%Y-%m-%d") -
                     datetime.strptime(state["start_date"], "%Y-%m-%d")).days
        dest = state["destination"]
        origin = state["origin"]

        flights = state.get("flights") or {}
        onward = flights.get("onward") or {}
        return_flight = flights.get("return") or {}
        dep_time = onward.get("departure", "14:30")
        arr_time = onward.get("arrival", "17:45")
        ret_dep = return_flight.get("departure", "19:00")
        ret_arr = return_flight.get("arrival", "22:15")

        restaurants = state.get("restaurants") or []
        sightseeing = state.get("sightseeing") or []
        activities = state.get("activities") or []

        def r_name(i: int) -> str:
            return restaurants[i]["name"] if i < len(restaurants) else "Nearby restaurant"

        def s_name(i: int) -> str:
            return sightseeing[i]["name"] if i < len(sightseeing) else "Local attraction"

        def s_time(i: int) -> str:
            return sightseeing[i].get("time_needed", "2 hours") if i < len(sightseeing) else "2 hours"

        def a_name(i: int) -> str:
            return activities[i]["name"] if i < len(activities) else "Booked activity"

        state["itinerary"] = []

        # Day 1 - Arrival
        state["itinerary"].append({
            "day": 1,
            "title": "Arrival Day",
            "schedule": [
                {"time": dep_time, "activity": f"✈️ Flight departure from {origin}"},
                {"time": arr_time, "activity": f"🛬 Arrival at {dest}"},
                {"time": "18:30", "activity": "🏨 Check-in at hotel, fresh up"},
                {"time": "20:00", "activity": f"🍽️ Dinner at {r_name(0)} (nearby)"},
                {"time": "21:30", "activity": "😴 Rest at hotel, early sleep (next day busy!)"},
            ],
        })

        # Middle days: distribute sightseeing and activities using actual names from state
        sight_idx = 0
        rest_idx = 1
        act_idx = 0
        middle_days = max(1, days_total - 1)

        for d in range(1, middle_days + 1):
            day_num = d + 1
            schedule = [
                {"time": "08:00", "activity": "☕ Breakfast at hotel"},
            ]
            # Morning: 1–2 sightseeing spots
            if sight_idx < len(sightseeing):
                schedule.append({"time": "09:00", "activity": f"🚗 Travel to {s_name(sight_idx)}"})
                schedule.append({"time": "09:30-12:00", "activity": f"🏛️ {s_name(sight_idx)} visit ({s_time(sight_idx)})"})
                sight_idx += 1
            # Lunch from state
            schedule.append({"time": "12:30", "activity": f"🍽️ Lunch at {r_name(rest_idx)}"})
            rest_idx += 1
            # Afternoon: optional second sight or activity
            if sight_idx < len(sightseeing):
                schedule.append({"time": "14:00", "activity": f"🚗 Travel to {s_name(sight_idx)}"})
                schedule.append({"time": "14:30-16:30", "activity": f"🏰 {s_name(sight_idx)} exploration"})
                sight_idx += 1
            elif act_idx < len(activities):
                schedule.append({"time": "14:00-17:00", "activity": f"🎫 {a_name(act_idx)} (booked activity)"})
                act_idx += 1
            else:
                schedule.append({"time": "14:00", "activity": "☕ Tea/Coffee break at cafe"})
            # Evening
            schedule.append({"time": "18:00-20:00", "activity": "🛍️ Local area / market (shopping or stroll)"})
            schedule.append({"time": "20:30", "activity": f"🍽️ Dinner at {r_name(rest_idx)}"})
            rest_idx += 1
            schedule.append({"time": "22:00", "activity": "🏨 Back to hotel, rest"})

            title = "Major Attractions Day" if d == 1 else ("Experiences & Leisure" if d == 2 else f"Day {day_num} – Sightseeing & Food")
            state["itinerary"].append({"day": day_num, "title": title, "schedule": schedule})

        # Last day - Departure
        state["itinerary"].append({
            "day": days_total + 1,
            "title": "Departure Day",
            "schedule": [
                {"time": "08:00", "activity": "☕ Breakfast at hotel"},
                {"time": "10:00", "activity": "🏨 Check-out from hotel"},
                {"time": "11:00", "activity": "🛍️ Last minute shopping (if time)"},
                {"time": "16:00", "activity": "🚗 Leave for airport (3 hours before flight)"},
                {"time": ret_dep, "activity": f"✈️ Flight to {origin}"},
                {"time": ret_arr, "activity": "🏠 Reach home sweet home!"},
            ],
        })

        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"✅ {len(state['itinerary'])} din ka complete schedule ready!"}
        ]
        print("✅ Itinerary ready!")
        return state

    def _to_float(self, x, default: float = 0.0) -> float:
        """Coerce LLM output to float (they often return numbers as strings)."""
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            try:
                return float(x.replace(",", "").strip())
            except (ValueError, TypeError):
                return default
        return default

    def calculate_budget(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        calculate the total budget for the trip
        - flights, hotels, local transport, restaurants, sightseeing, activities
        """
        print("Calculating budget...")
        
        days = (datetime.strptime(state['end_date'], '%Y-%m-%d') - 
                datetime.strptime(state['start_date'], '%Y-%m-%d')).days
        travelers = max(1, int(self._to_float(state.get("travelers", 1), 1)))
        
        flights = state.get("flights") or {}
        flight_cost = self._to_float(flights.get("total_cost"), 0) * travelers
        hotels = state.get("hotels") or []
        # Prefer mid-range (index 1), else first hotel
        if len(hotels) > 1:
            hotel_cost = self._to_float(hotels[1].get("total_cost"), 0)
        elif hotels:
            hotel_cost = self._to_float(hotels[0].get("total_cost"), 0)
        else:
            hotel_cost = 0.0
        
        # Local transport
        lt = state.get("local_transport") or {}
        primary = lt.get("primary") or {}
        transport_cost = self._to_float(primary.get("estimated_daily_cost"), 800) * days

        # Food (3 meals per day)
        food_cost = (300 + 500 + 600) * days * travelers

        # Sightseeing (entry_fee can be string from LLM)
        sight_list = state.get("sightseeing") or []
        sightseeing_cost = sum(self._to_float(p.get("entry_fee"), 0) for p in sight_list) * travelers

        # Activities (cost can be string from LLM)
        act_list = state.get("activities") or []
        activities_cost = sum(self._to_float(a.get("cost"), 0) for a in act_list[:2]) * travelers
        
        # Shopping budget
        shopping_budget = 3000 * travelers
        
        # Miscellaneous (20% buffer)
        subtotal = (flight_cost + hotel_cost + transport_cost + food_cost + 
                   sightseeing_cost + activities_cost + shopping_budget)
        misc_buffer = subtotal * 0.20
        
        total_estimated = subtotal + misc_buffer
        budget_available = self._to_float(state.get("budget_total"), 0)
        
        state["budget_breakdown"] = {
            "flights": flight_cost,
            "hotel": hotel_cost,
            "local_transport": transport_cost,
            "food": food_cost,
            "sightseeing": sightseeing_cost,
            "activities": activities_cost,
            "shopping": shopping_budget,
            "miscellaneous": misc_buffer,
            "total_estimated": total_estimated,
            "budget_available": budget_available,
            "surplus_or_deficit": budget_available - total_estimated,
            "per_person": total_estimated / travelers
        }
        
        status = "✅ Budget ke andar!" if budget_available >= total_estimated else "⚠️ Budget thoda over!"
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"{status} Total: ₹{total_estimated:.0f}"}
        ]
        
        print("✅ Budget calculated!")
        return state

    def generate_smart_tips(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        generate smart tips based on itinerary and activities
        - save money, avoid crowds, etc.
        """
        print("Generating smart tips...")
        
        weather_gen = state.get("weather_forecast", {}).get("general", "Moderate")
        duration_days = (datetime.strptime(state['end_date'], '%Y-%m-%d') - datetime.strptime(state['start_date'], '%Y-%m-%d')).days
        
        prompt = f"""
        Generate smart travel tips for {state['destination']} trip.
        
        Based on:
        - Weather: {weather_gen}
        - Budget: ₹{state['budget_total']}
        - Duration: {duration_days} days
        - Travelers: {state['travelers']}
        
        Provide:
        1. Money-saving hacks
        2. Safety tips
        3. Local etiquette
        4. Best practices
        5. What to avoid
        
        Return JSON: {{"tips": ["string1", "string2", ...], "warnings": ["string1", ...]}}
        Tips format: Category header like "💰 MONEY TIPS:" then "  • tip text" - destination-specific, practical.
        """
        
        fallback_result = {
            "tips": [
                "💰 MONEY TIPS:", "  • Use Metro/Uber wisely", "  • Street food + fancy dinner balance",
                "🔒 SAFETY TIPS:", "  • Valuables hotel safe", "  • Emergency contacts save karo",
                "🎯 LOCAL HACKS:", "  • Morning monuments best", "  • Bargaining in markets",
                "⚠️ AVOID:", "  • Peak hours travel", "  • Unmetered autos (price fix karo)"
            ],
            "warnings": ["🚨 Document photocopies", "🚨 Travel insurance", "🚨 Hotel address card", "🚨 Medicine kit"]
        }
        result = self._call_llm_json(prompt, '{"tips":["..."],"warnings":["..."]}', fallback_result)
        state["tips"] = result.get("tips", fallback_result["tips"]) if isinstance(result, dict) else fallback_result["tips"]
        state["warnings"] = result.get("warnings", fallback_result["warnings"]) if isinstance(result, dict) else fallback_result["warnings"]
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": f"✅ {len(state['tips'])} smart tips ready!"}
        ]
        
        print("✅ Tips generated!")
        return state
    
    def finalize_complete_plan(self, state:CompleteTravelState) -> CompleteTravelState:
        """
        finalize the complete travel plan
        - summary, recommendations, next steps
        """
        print("Finalizing complete plan...")
        
        bb = state.get("budget_breakdown", {})
        top_places = [p["name"] for p in state.get("sightseeing", [])[:3]]
        top_food = [r["name"] for r in state.get("restaurants", [])[:2]]
        
        prompt = f"""
        Trip Summary for LLM:
        - Destination: {state.get('destination', '')}
        - Duration: {len(state.get('itinerary', []))} days, {state.get('travelers', 1)} travelers
        - Total cost: ₹{bb.get('total_estimated', 0):.0f}, Per person: ₹{bb.get('per_person', 0):.0f}
        - Top places: {', '.join(top_places) if top_places else 'N/A'}
        - Top restaurants: {', '.join(top_food) if top_food else 'N/A'}
        - {len(state.get('sightseeing', []))} places, {len(state.get('restaurants', []))} restaurants
        - Budget status: {'within budget' if bb.get('surplus_or_deficit', 0) >= 0 else 'slightly over'}
        
        Generate 6-8 personalized final recommendations/summary points for this trip.
        Hinglish mein, practical aur encouraging. Format: array of strings, each starting with ✅.
        Return JSON: ["✅ ...", "✅ ...", ...]
        """
        fallback_recs = [
            f"✅ Total trip cost: ₹{bb.get('total_estimated', 0):.0f}",
            f"✅ Per person: ₹{bb.get('per_person', 0):.0f}",
            f"✅ {len(state.get('sightseeing', []))} places to visit",
            f"✅ {len(state.get('restaurants', []))} restaurants recommended",
            f"✅ {len(state.get('itinerary', []))} days fully planned",
            "✅ Packing list ready",
            "✅ Weather forecast checked",
            "✅ Safety tips included"
        ]
        recs = self._call_llm_json(prompt, '["✅ ...", "✅ ..."]', fallback_recs)
        state["recommendations"] = recs if isinstance(recs, list) and recs else fallback_recs
        
        state["current_step"] = "completed"
        
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": "🎉 Complete travel plan ready! Bas bags pack karo aur nikal pado!"}
        ]
        
        print("✅ PLAN COMPLETE!")
        return state
        
    def plan_trip(self, trip_details: Dict) -> CompleteTravelState:
        """
        plan the trip based on the trip data
        Input format:
        {
            "origin": "Delhi",
            "destination": "Goa",
            "start_date": "2026-04-01",
            "end_date": "2026-04-05",
            "travelers": 2,
            "budget_total": 50000,
            "preferences": {
                "type": "leisure",  # adventure, cultural, party, family
                "pace": "relaxed"   # relaxed, moderate, packed
            }
        }
        """
        initial_state = CompleteTravelState(
            origin=trip_details["origin"],
            destination=trip_details["destination"],
            start_date=trip_details["start_date"],
            end_date=trip_details["end_date"],
            travelers=trip_details.get("travelers", 1),
            budget_total=trip_details.get("budget_total", 50000),
            preferences=trip_details.get("preferences", {"type": "leisure", "pace": "relaxed"}),
            flights={},
            hotels=[],
            local_transport={},
            restaurants=[],
            sightseeing=[],
            activities=[],
            weather_forecast={},
            packing_list=[],
            itinerary=[],
            budget_breakdown={},
            recommendations=[],
            tips=[],
            warnings=[],
            messages=[],
            current_step="started"
        )
        
        print("\n" + "="*70)
        print("🚀 COMPLETE TRAVEL PLANNING SHURU!")
        print("="*70 + "\n")
        
        # LangGraph workflow execute karte hain
        final_state = self.graph.invoke(initial_state)
        
        return final_state

    

        
        
        
        

