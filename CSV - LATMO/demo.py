"""
Demo script to test the Timetable Agent
"""

from agent import TimetableAgent
import sys

def main():
    print("Initializing Timetable Agent...")
    agent = TimetableAgent()
    
    if len(sys.argv) > 1:
        # If command-line argument provided, use it as the query
        query = " ".join(sys.argv[1:])
    else:
        # Otherwise, enter interactive mode
        print("\nTimetable Query Assistant")
        print("=========================")
        print("Type 'exit' to quit")
        print()
        
        chat_history = []
        
        while True:
            query = input("Your question: ")
            
            if query.lower() in ["exit", "quit"]:
                break
                
            try:
                response = agent.query(query, chat_history)
                print(f"\nAnswer: {response}\n")
                
                # Add to chat history
                chat_history.append({"type": "human", "content": query})
                chat_history.append({"type": "ai", "content": response})
                
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
