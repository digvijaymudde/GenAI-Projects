
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from typing import Annotated
from typing_extensions import TypedDict
import operator

class TravelState(TypedDict, total=False):
    origin: str
    destination: str
    days: int
    travelers: int
    budget: float

    travel_options: list[dict]
    weather: dict
    hotel_options: list[dict]

    # Merge errors from parallel nodes instead of overwriting.
    errors: Annotated[list[str], operator.add]

    final_plan: str



def run_with_retries(operation, attempts=3):
    """Return (result, error). Retry failures up to a fixed limit."""
    last_error = None

    for attempt in range(attempts):
        try:
            return operation(), None
        except Exception as exc:
            last_error = exc

    return None, str(last_error)


def search_travel(state: TravelState):
    def operation():
        # Replace with a real travel API call.
        return [
            {"mode": "Bus", "price": 1800},
            {"mode": "Train", "price": 1200},
        ]

    result, error = run_with_retries(operation)

    if error:
        return {
            "travel_options": [],
            "errors": [f"Travel search failed: {error}"],
        }

    return {"travel_options": result}


def check_weather(state: TravelState):
    def operation():
        # Replace with a real weather API call.
        # Mock success for now.
        return {
            "summary": "Example weather result",
            "source": "mock",
        }

    result, error = run_with_retries(operation)

    if error:
        return {
            "weather": {
                "status": "unavailable",
                "summary": None,
            },
            "errors": [f"Weather lookup failed: {error}"],
        }

    return {
        "weather": {
            "status": "available",
            **result,
        }
    }


def search_hotels(state: TravelState):
    def operation():
        # Replace with a real hotel API call.
        return [
            {"name": "Budget Stay", "nightly_price": 2500},
            {"name": "Coastal Hotel", "nightly_price": 4000},
        ]

    result, error = run_with_retries(operation)

    if error:
        return {
            "hotel_options": [],
            "errors": [f"Hotel search failed: {error}"],
        }

    return {"hotel_options": result}




def combine_results(state: TravelState):
    travel = state.get("travel_options", [])
    hotels = state.get("hotel_options", [])
    weather = state.get("weather", {})
    errors = state.get("errors", [])

    lines = [
        f"Trip: {state['origin']} to {state['destination']}",
        f"Duration: {state['days']} days",
        f"Travelers: {state['travelers']}",
        f"Budget: Rs. {state['budget']}",
        "",
        f"Travel options: {travel or 'Unavailable'}",
        f"Hotel options: {hotels or 'Unavailable'}",
    ]

    if weather.get("status") == "available":
        lines.append(f"Weather: {weather.get('summary')}")
    else:
        lines.append(
            "Weather: Unavailable; check a reliable forecast before travelling."
        )

    if errors:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {error}" for error in errors)

    return {"final_plan": "\n".join(lines)}


builder = StateGraph(TravelState)

builder.add_node("validate", validate_request)
builder.add_node("search_travel", search_travel)
builder.add_node("check_weather", check_weather)
builder.add_node("search_hotels", search_hotels)
builder.add_node("combine", combine_results)

builder.add_edge(START, "validate")

builder.add_edge("validate", "search_travel")
builder.add_edge("validate", "check_weather")
builder.add_edge("validate", "search_hotels")

# Fan-in waits for all three upstream branches.
builder.add_edge(
    ["search_travel", "check_weather", "search_hotels"],
    "combine",
)

builder.add_edge("combine", END)

graph = builder.compile()

result = graph.invoke({
    "origin": "Pune",
    "destination": "Goa",
    "days": 3,
    "travelers": 2,
    "budget": 25000,
    "errors": [],
})

print(result["final_plan"])
print("Captured errors:", result.get("errors", []))
