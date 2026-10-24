from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


# ---------------------------------------------------------------------------
# 1. Weather Subgraph Definitions
# ---------------------------------------------------------------------------

class WeatherCallState(TypedDict):
    destination: str
    weather: dict


def fetch_primary_weather(state: WeatherCallState):
    """
    Mock primary weather API.
    Raises TimeoutError to trigger RetryPolicy and fallback.
    To test the success path, return:
    return {"weather": {"status": "available", "summary": "Sunny, 28°C"}}
    """
    print("[Primary Weather API] Attempting connection... (Timing out)")
    raise TimeoutError("Primary weather API connection timed out.")


# Inner graph: LangGraph manages automated retries here
inner_weather_builder = StateGraph(WeatherCallState)
inner_weather_builder.add_node(
    "fetch",
    fetch_primary_weather,
    retry_policy=RetryPolicy(max_attempts=3),
)
inner_weather_builder.add_edge(START, "fetch")
inner_weather_builder.add_edge("fetch", END)
primary_weather_inner_graph = inner_weather_builder.compile()


class WeatherState(TypedDict, total=False):
    destination: str
    weather: dict
    weather_error: str
    weather_source: str


def primary_weather_node(state: WeatherState):
    """Invokes inner graph with retries; converts exhausted failures into state."""
    try:
        result = primary_weather_inner_graph.invoke({
            "destination": state["destination"],
            "weather": {},
        })
        return {
            "weather": result["weather"],
            "weather_error": "",
            "weather_source": "primary",
        }
    except Exception as exc:
        print(f"[Primary Weather] Retries exhausted. Error captured: {exc}")
        return {
            "weather": {"status": "unavailable"},
            "weather_error": str(exc),
            "weather_source": "",
        }


def fallback_weather_node(state: WeatherState):
    """Alternative provider triggered only when primary fails."""
    print("[Fallback Weather API] Querying backup service...")
    try:
        # Mock fallback success (replace with backup vendor/cache lookup)
        result = {
            "status": "available",
            "summary": "Clear skies, 29°C (via backup radar)",
        }
        return {
            "weather": result,
            "weather_error": "",
            "weather_source": "fallback",
        }
    except Exception as exc:
        return {
            "weather": {"status": "unavailable"},
            "weather_error": f"Fallback also failed: {exc}",
            "weather_source": "",
        }


def mark_weather_unavailable(state: WeatherState):
    """Graceful degradation state when both primary and fallback fail."""
    print("[Weather Status] Weather marked as unavailable.")
    return {
        "weather": {"status": "unavailable", "summary": None},
        "weather_source": "none",
    }


# Routing logic for Weather Subgraph
def route_after_primary(state: WeatherState):
    if state.get("weather", {}).get("status") == "available":
        return "done"
    return "fallback"


def route_after_fallback(state: WeatherState):
    if state.get("weather", {}).get("status") == "available":
        return "done"
    return "unavailable"


# Compile Weather Subgraph
weather_subgraph_builder = StateGraph(WeatherState)
weather_subgraph_builder.add_node("primary", primary_weather_node)
weather_subgraph_builder.add_node("fallback", fallback_weather_node)
weather_subgraph_builder.add_node("unavailable", mark_weather_unavailable)

weather_subgraph_builder.add_edge(START, "primary")
weather_subgraph_builder.add_conditional_edges(
    "primary",
    route_after_primary,
    {"done": END, "fallback": "fallback"},
)
weather_subgraph_builder.add_conditional_edges(
    "fallback",
    route_after_fallback,
    {"done": END, "unavailable": "unavailable"},
)
weather_subgraph_builder.add_edge("unavailable", END)

weather_workflow = weather_subgraph_builder.compile()


# ---------------------------------------------------------------------------
# 2. Main Travel Assistant Graph
# ---------------------------------------------------------------------------

class TravelState(TypedDict, total=False):
    origin: str
    destination: str
    travel_options: List[str]
    hotel_options: List[str]
    weather: dict
    weather_error: str
    weather_source: str
    final_plan: str


def validate_request(state: TravelState):
    print(f"\n--- Planning itinerary from {state['origin']} to {state['destination']} ---")
    return state


def search_travel(state: TravelState):
    return {
        "travel_options": [
            "Morning Flight: 1h 15m duration",
            "Overnight Sleeper Bus: 8h duration",
        ]
    }


def search_hotels(state: TravelState):
    return {
        "hotel_options": [
            "Sea View Resort & Spa (4.5★)",
            "Beachfront Boutique Stay (4.2★)",
        ]
    }


def combine_results(state: TravelState):
    lines = [
        f"Trip Overview: {state['origin']} -> {state['destination']}",
        "\nTravel Options:",
    ]
    for opt in state.get("travel_options", []):
        lines.append(f" - {opt}")

    lines.append("\nHotel Recommendations:")
    for hotel in state.get("hotel_options", []):
        lines.append(f" - {hotel}")

    lines.append("\nWeather Forecast:")
    weather_status = state.get("weather", {}).get("status")
    if weather_status == "available":
        lines.append(f" - Conditions: {state['weather'].get('summary')}")
        if state.get("weather_source") == "fallback":
            lines.append(" - Note: Forecast obtained via secondary weather provider.")
    else:
        lines.append(" - Notice: Live weather updates could not be verified.")
        lines.append("   Please check local forecasts before your trip.")

    return {"final_plan": "\n".join(lines)}


# Main Graph Assembly
main_builder = StateGraph(TravelState)

main_builder.add_node("validate", validate_request)
main_builder.add_node("search_travel", search_travel)
main_builder.add_node("weather_workflow", weather_workflow)
main_builder.add_node("search_hotels", search_hotels)
main_builder.add_node("combine", combine_results)

# Fan-out: parallel execution
main_builder.add_edge(START, "validate")
main_builder.add_edge("validate", "search_travel")
main_builder.add_edge("validate", "weather_workflow")
main_builder.add_edge("validate", "search_hotels")

# Fan-in: wait for all branches before final synthesis
main_builder.add_edge(
    ["search_travel", "weather_workflow", "search_hotels"],
    "combine",
)
main_builder.add_edge("combine", END)

travel_app = main_builder.compile()


# ---------------------------------------------------------------------------
# Execution Test Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    initial_input = {
        "origin": "Pune",
        "destination": "Goa",
    }
    result = travel_app.invoke(initial_input)
    print("\n================ FINAL GENERATED PLAN ================")
    print(result["final_plan"])
    print("======================================================")