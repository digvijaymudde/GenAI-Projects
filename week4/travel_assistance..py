
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. Define the shared state
class TravelState(TypedDict, total=False):
    origin: str
    destination: str
    days: int
    travelers: int
    budget: float

    travel_options: list[dict]
    weather: dict
    hotel_options: list[dict]

    errors: list[str]
    final_plan: str


# 2. Define graph nodes
def validate_request(state: TravelState):
    if not state.get("origin") or not state.get("destination"):
        raise ValueError("Origin and destination are required")

    if state.get("days", 0) < 1:
        raise ValueError("Trip duration must be at least one day")

    if state.get("travelers", 0) < 1:
        raise ValueError("At least one traveler is required")

    if state.get("budget", 0) <= 0:
        raise ValueError("Budget must be greater than zero")

    return {}


def search_travel(state: TravelState):
    # Mock data: later we will connect a real travel API.
    return {
        "travel_options": [
            {"mode": "Bus", "price": 1800},
            {"mode": "Train", "price": 1200},
        ]
    }


def check_weather(state: TravelState):
    # Mock data: not a real forecast.
    return {
        "weather": {
            "summary": "Example weather result",
            "source": "mock"
        }
    }


def search_hotels(state: TravelState):
    # Mock data: later we will connect a hotel API.
    return {
        "hotel_options": [
            {"name": "Budget Stay", "nightly_price": 2500},
            {"name": "Coastal Hotel", "nightly_price": 4000},
        ]
    }


def combine_results(state: TravelState):
    travel = state.get("travel_options", [])
    hotels = state.get("hotel_options", [])
    weather = state.get("weather", {})

    return {
        "final_plan": (
            f"Trip: {state['origin']} to {state['destination']}\n"
            f"Duration: {state['days']} days\n"
            f"Travel options: {travel}\n"
            f"Hotel options: {hotels}\n"
            f"Weather: {weather}\n"
            f"Budget: ₹{state['budget']}"
        )
    }


# 3. Build the graph
builder = StateGraph(TravelState)

builder.add_node("validate", validate_request)
builder.add_node("search_travel", search_travel)
builder.add_node("check_weather", check_weather)
builder.add_node("search_hotels", search_hotels)
builder.add_node("combine", combine_results)

builder.add_edge(START, "validate")

# Fan-out: independent tasks
builder.add_edge("validate", "search_travel")
builder.add_edge("validate", "check_weather")
builder.add_edge("validate", "search_hotels")

# Fan-in: wait for all three tasks before combining
builder.add_edge(
    ["search_travel", "check_weather", "search_hotels"],
    "combine"
)

builder.add_edge("combine", END)

graph = builder.compile()


# 4. Run the graph
result = graph.invoke({
    "origin": "Pune",
    "destination": "Goa",
    "days": 3,
    "travelers": 2,
    "budget": 25000,
    "errors": [],
})

print(result["final_plan"])