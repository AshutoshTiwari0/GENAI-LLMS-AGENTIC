from tools.tavily_tool import tavily_search 
from tools.flight_tool import search_flights
from backend import run_travel_agent
# res=tavily_search("Best hotels in india")
# print(res)
# res=search_flights("Flights from New York to London")
res=run_travel_agent("I want to plan a trip to Paris, France. I need a flight from New York City and a hotel for 3 nights.")
print(res)