import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from groq import Groq

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# 1. WEATHER SUBGRAPH (Retries + Fallback)
# ============================================================

class WeatherCallState(TypedDict):
    destination: str
    weather: dict


def fetch_primary_weather(state: WeatherCallState):
    # Simulated transient API failure for testing retry & fallback
    raise TimeoutError("Primary weather API connection timed out")


# Inner graph: handles automatic retries
inner_builder = StateGraph(WeatherCallState)
inner_builder.add_node(
    "fetch",
    fetch_primary_weather,
    retry_policy=RetryPolicy(max_attempts=3),
)
inner_builder.add_edge(START, "fetch")
inner_builder.add_edge("fetch", END)
inner_weather_graph = inner_builder.compile()


class WeatherState(TypedDict, total=False):
    destination: str
    weather: dict
    weather_source: str


def primary_weather_node(state: WeatherState):
    try:
        res = inner_weather_graph.invoke({
            "destination": state["destination"],
            "weather": {},
        })
        return {"weather": res["weather"], "weather_source": "primary"}
    except Exception:
        return {"weather": {"status": "unavailable"}, "weather_source": ""}


def fallback_weather_node(state: WeatherState):
    return {
        "weather": {
            "status": "available",
            "summary": "28°C, Clear skies (Secondary radar)",
        },
        "weather_source": "fallback",
    }


def mark_weather_unavailable(state: WeatherState):
    return {
        "weather": {"status": "unavailable", "summary": None},
        "weather_source": "none",
    }


def route_after_primary(state: WeatherState):
    if state.get("weather", {}).get("status") == "available":
        return "done"
    return "fallback"


def route_after_fallback(state: WeatherState):
    if state.get("weather", {}).get("status") == "available":
        return "done"
    return "unavailable"


subgraph_builder = StateGraph(WeatherState)
subgraph_builder.add_node("primary", primary_weather_node)
subgraph_builder.add_node("fallback", fallback_weather_node)
subgraph_builder.add_node("unavailable", mark_weather_unavailable)

subgraph_builder.add_edge(START, "primary")
subgraph_builder.add_conditional_edges(
    "primary",
    route_after_primary,
    {"done": END, "fallback": "fallback"},
)
subgraph_builder.add_conditional_edges(
    "fallback",
    route_after_fallback,
    {"done": END, "unavailable": "unavailable"},
)
subgraph_builder.add_edge("unavailable", END)

weather_subgraph = subgraph_builder.compile()


# ============================================================
# 2. MAIN TRAVEL GRAPH
# ============================================================

class TravelState(TypedDict, total=False):
    raw_query: str
    origin: str
    destination: str
    is_valid: bool
    travel_options: List[str]
    hotel_options: List[str]
    weather: dict
    weather_source: str
    final_itinerary: str


def validate_request(state: TravelState):
    prompt = f"""Extract the origin and destination from this travel query.
Query: "{state['raw_query']}"

Respond ONLY with JSON:
{{"origin": "city or empty", "destination": "city or empty", "is_valid": true/false}}"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    is_valid = bool(data.get("origin") and data.get("destination"))
    return {
        "origin": data.get("origin", ""),
        "destination": data.get("destination", ""),
        "is_valid": is_valid,
    }


def route_validation(state: TravelState):
    return "valid" if state.get("is_valid") else "invalid"


def rejection_node(state: TravelState):
    return {
        "final_itinerary": "Please specify both a starting point and a destination (e.g., 'Trip from Pune to Goa')."
    }


def plan_trip_coordinator(state: TravelState):
    # Passthrough node to coordinate parallel fan-out
    return state


def search_travel_options(state: TravelState):
    return {
        "travel_options": [
            f"Direct flight from {state['origin']} to {state['destination']} (1h 10m)",
            f"Overnight train from {state['origin']} to {state['destination']} (8h)",
        ]
    }


def search_hotels(state: TravelState):
    return {
        "hotel_options": [
            f"Coastal Beach Resort ({state['destination']}) - 4.5★",
            f"Boutique Heritage Hotel ({state['destination']}) - 4.2★",
        ]
    }


def synthesize_itinerary(state: TravelState):
    prompt = f"""Create a clear travel summary using these details:
- Trip: {state['origin']} to {state['destination']}
- Transport: {state.get('travel_options')}
- Stays: {state.get('hotel_options')}
- Weather: {state.get('weather', {}).get('summary', 'Unavailable')} (Source: {state.get('weather_source')})

Note: If weather is unavailable or from fallback, include a brief advisory note."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {"final_itinerary": response.choices[0].message.content}


# ============================================================
# 3. GRAPH ASSEMBLY
# ============================================================

main_builder = StateGraph(TravelState)

# Register nodes
main_builder.add_node("validate", validate_request)
main_builder.add_node("plan_trip", plan_trip_coordinator)
main_builder.add_node("rejected", rejection_node)
main_builder.add_node("travel_search", search_travel_options)
main_builder.add_node("hotel_search", search_hotels)
main_builder.add_node("weather_service", weather_subgraph)
main_builder.add_node("synthesize", synthesize_itinerary)

# Routing setup
main_builder.add_edge(START, "validate")

main_builder.add_conditional_edges(
    "validate",
    route_validation,
    {"valid": "plan_trip", "invalid": "rejected"},
)

# Parallel fan-out
main_builder.add_edge("plan_trip", "travel_search")
main_builder.add_edge("plan_trip", "hotel_search")
main_builder.add_edge("plan_trip", "weather_service")

# Parallel fan-in
main_builder.add_edge(
    ["travel_search", "hotel_search", "weather_service"],
    "synthesize",
)

main_builder.add_edge("synthesize", END)
main_builder.add_edge("rejected", END)

app = main_builder.compile()


# ============================================================
# 4. EXECUTION
# ============================================================

if __name__ == "__main__":
    test_query = "Plan a vacation from Pune to Goa next Friday"
    print(f"Running query: '{test_query}'\n")

    result = app.invoke({"raw_query": test_query})

    print("=" * 60)
    print(result["final_itinerary"])
    print("=" * 60)