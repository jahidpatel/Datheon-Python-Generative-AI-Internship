
from agent import run_agent


print("\n======================================")
print("       AUTONOMOUS AI AGENT TEST")
print("======================================")


# TEST 1: Calculator tool
print("\nTEST 1: Calculator")
run_agent("Calculate 25% of 8400.")


# TEST 2: Web search tool
print("\nTEST 2: Web Search")
run_agent("Search the web for the latest developments in Generative AI.")


# TEST 3: Local file reader
print("\nTEST 3: Document Reader")
run_agent(
    "Read urban_gardening_and_sustainable_cities.txt "
    "and explain its main points."
)


print("\n======================================")
print("          TESTING FINISHED")
print("======================================")
