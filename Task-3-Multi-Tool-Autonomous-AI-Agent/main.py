
from agent import run_agent


def main():

    print("\n" + "=" * 55)
    print("       MULTI-TOOL AUTONOMOUS AI AGENT")
    print("=" * 55)

    print("\nWelcome! I am your AI assistant.")
    print("I can search the web, calculate and read documents.")

    print("\nAvailable tools:")
    print("1. Web Search")
    print("2. Python Calculator")
    print("3. Local PDF/TXT File Reader")

    print("\nType 'exit' or 'quit' to close the application.")

    while True:

        print("\n" + "-" * 55)

        user_query = input("\nYou: ").strip()

        if not user_query:
            print("Please enter a question.")
            continue

        if user_query.lower() in ["exit", "quit"]:
            print("\nThank you for using the AI Agent!")
            break

        run_agent(user_query)


if __name__ == "__main__":
    main()
