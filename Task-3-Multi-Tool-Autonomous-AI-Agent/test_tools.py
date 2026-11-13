
from tools import web_search, calculator, read_local_file


print("\n========== AI AGENT TOOL TESTING ==========\n")


# TEST 1: WEB SEARCH
print("TEST 1: WEB SEARCH")
print("------------------------------------------")

result = web_search.invoke({
    "query": "Latest developments in Artificial Intelligence"
})

print(result)


# TEST 2: CALCULATOR
print("\n\nTEST 2: PYTHON CALCULATOR")
print("------------------------------------------")

result = calculator.invoke({
    "expression": "(50000 * 15) / 100"
})

print(result)


# TEST 3: LOCAL FILE READER
print("\n\nTEST 3: LOCAL FILE READER")
print("------------------------------------------")

result = read_local_file.invoke({
    "filename": "urban_gardening_and_sustainable_cities.txt"
})

print(result)


print("\n========== TESTING COMPLETED ==========\n")
