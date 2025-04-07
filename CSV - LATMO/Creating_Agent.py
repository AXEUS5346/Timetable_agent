"""
TimetableCreatorAgent for creating and modifying personalized timetables
Uses Groq models for LLM-based decision making
Schema-independent and flexible attribute support
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
import os
import json
import csv
import logging
from datetime import datetime
import argparse
from typing import List, Dict, Any, Optional, Set, Union
import re

# Import Accessing_Agent for database queries
from Accessing_Agent import TimetableAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
TIMETABLES_PATH = os.path.join(DATABASE_PATH, "timetables")
SCHEMA_FILE = os.path.join(TIMETABLES_PATH, "_schema.json")
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # Default Groq model

class TimetableCreatorAgent:
    """
    Agent for creating and modifying personalized timetables with flexible attributes
    Completely schema-independent and adapts to user preferences
    """
    
    def __init__(self, groq_model: str = DEFAULT_GROQ_MODEL, groq_api_key: Optional[str] = None):
        """Initialize the agent with the specified Groq model."""
        logger.info(f"Initializing TimetableCreatorAgent with Groq model: {groq_model}")
        
        # Check for API key in environment variables if not provided
        if groq_api_key is None:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                logger.warning("No Groq API key found. Please provide an API key through the --api-key parameter or set the GROQ_API_KEY environment variable.")
        
        try:
            # Initialize Groq LLM
            self.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=groq_model,
                temperature=0.2
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize TimetableAgent for querying data
            self.query_agent = TimetableAgent()
            
            self.schemas = self._load_schemas()
            
            # Create the schema file if it doesn't exist
            if not os.path.exists(SCHEMA_FILE):
                logger.info("No schema file found. Creating a new one.")
                self._save_schemas({"default": {
                    "attributes": [
                        "timetable_id", "version", "timestamp", 
                        "course", "room", "time", "day", 
                        "instructor", "created_by"
                    ],
                    "required": ["timetable_id", "course"]
                }})
                
            # Discover available CSV files
            self.available_csv_files = self._discover_csv_files()
            
            # Extract schema for each CSV file that's not in the schema file
            for csv_file in self.available_csv_files:
                csv_name = os.path.basename(csv_file).replace('.csv', '')
                if csv_name not in self.schemas and csv_name != "_schema":
                    self._extract_schema_from_csv(csv_file)
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def _discover_csv_files(self) -> List[str]:
        """Discover all CSV files in the timetables directory."""
        csv_files = []
        for file in os.listdir(TIMETABLES_PATH):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(TIMETABLES_PATH, file))
        logger.info(f"Discovered {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        return csv_files
    
    def _extract_schema_from_csv(self, csv_path: str) -> None:
        """Extract schema from a CSV file and add it to the schemas."""
        try:
            # Read the CSV header
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            
            if headers:
                # Get the filename without extension
                filename = os.path.basename(csv_path).replace('.csv', '')
                
                # Add to schemas
                self.schemas[filename] = {
                    "attributes": headers,
                    "required": [headers[0]] if headers else []  # First column is typically a primary key
                }
                
                logger.info(f"Extracted schema for '{filename}': {headers}")
                self._save_schemas(self.schemas)
        except Exception as e:
            logger.error(f"Error extracting schema from {csv_path}: {str(e)}")
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load schema definitions from the schema file."""
        if os.path.exists(SCHEMA_FILE):
            try:
                with open(SCHEMA_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading schema file: {str(e)}")
                return {}
        return {}
    
    def _save_schemas(self, schemas: Dict[str, Any]) -> None:
        """Save schema definitions to the schema file."""
        try:
            with open(SCHEMA_FILE, 'w') as f:
                json.dump(schemas, f, indent=2)
            self.schemas = schemas
        except Exception as e:
            logger.error(f"Error saving schema file: {str(e)}")
            raise
    
    def _get_csv_path(self, schema_name: str) -> str:
        """Get the path to a CSV file for a specific schema."""
        return os.path.join(TIMETABLES_PATH, f"{schema_name}.csv")
    
    def _load_csv_data(self, schema_name: str) -> List[Dict[str, Any]]:
        """Load data from a CSV file."""
        csv_path = self._get_csv_path(schema_name)
        if not os.path.exists(csv_path):
            return []
        
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            logger.error(f"Error loading CSV data from {csv_path}: {str(e)}")
            return []
    
    def _save_csv_data(self, schema_name: str, data: List[Dict[str, Any]]) -> None:
        """Save data to a CSV file."""
        csv_path = self._get_csv_path(schema_name)
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            
            # Get all field names from all rows
            fieldnames = set()
            for row in data:
                fieldnames.update(row.keys())
            
            # Ensure field names are in the same order as the schema if possible
            if schema_name in self.schemas:
                schema_fields = self.schemas[schema_name]["attributes"]
                # Order fields according to schema, with any extra fields at the end
                ordered_fields = [f for f in schema_fields if f in fieldnames]
                ordered_fields.extend([f for f in fieldnames if f not in schema_fields])
                fieldnames = ordered_fields
            
            # Write the data to CSV
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                writer.writeheader()
                writer.writerows(data)
        except Exception as e:
            logger.error(f"Error saving CSV data to {csv_path}: {str(e)}")
            raise
    
    def define_schema(self, schema_name: str, attributes: List[str], required: List[str]) -> None:
        """Define a new schema or update an existing one."""
        # Update the schemas
        schemas = self._load_schemas()
        schemas[schema_name] = {
            "attributes": attributes,
            "required": required
        }
        self._save_schemas(schemas)
        logger.info(f"Schema '{schema_name}' defined with attributes: {attributes}")
        
        # Create an empty CSV file if it doesn't exist
        csv_path = self._get_csv_path(schema_name)
        if not os.path.exists(csv_path):
            # Create with headers only
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(attributes)
    
    def get_available_schemas(self) -> Dict[str, Any]:
        """Get available schema definitions."""
        return self._load_schemas()
    
    def create_entry(self, schema_name: str, entry_data: Dict[str, Any]) -> bool:
        """
        Create a new entry in any CSV file with the given data.
        
        Args:
            schema_name: The name of the schema/file to use
            entry_data: The data for the new entry
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if schema exists
        schemas = self._load_schemas()
        if schema_name not in schemas:
            logger.error(f"Schema '{schema_name}' does not exist")
            return False
        
        # Validate required fields
        schema = schemas[schema_name]
        for field in schema["required"]:
            if field not in entry_data or not entry_data[field]:
                logger.error(f"Required field '{field}' is missing or empty")
                return False
        
        # Special handling for timetable entries (legacy support)
        if schema_name == "default" or schema_name == "timetable":
            # Add metadata fields if needed
            if "timestamp" in schema["attributes"] and "timestamp" not in entry_data:
                entry_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if "created_by" in schema["attributes"] and "created_by" not in entry_data:
                entry_data["created_by"] = "TimetableCreatorAgent"
            
            # Auto-increment timetable_id if needed
            if "timetable_id" in schema["attributes"] and "timetable_id" not in entry_data:
                data = self._load_csv_data(schema_name)
                ids = [int(row.get("timetable_id", 0)) for row in data if row.get("timetable_id", "").isdigit()]
                entry_data["timetable_id"] = str(max(ids + [0]) + 1)
            
            # Set version to 1 if needed
            if "version" in schema["attributes"] and "version" not in entry_data:
                entry_data["version"] = "1"
        
        # Auto-increment ID field if it's the first required field and not provided
        if not entry_data.get(schema["required"][0]) and "_id" in schema["required"][0]:
            id_field = schema["required"][0]
            data = self._load_csv_data(schema_name)
            ids = [int(row.get(id_field, 0)) for row in data if row.get(id_field, "").isdigit()]
            entry_data[id_field] = str(max(ids + [0]) + 1)
        
        # Load existing data
        data = self._load_csv_data(schema_name)
        
        # Add new entry
        data.append(entry_data)
        
        # Save updated data
        self._save_csv_data(schema_name, data)
        logger.info(f"Created new entry in '{schema_name}': {entry_data}")
        
        return True
    
    def update_entry(self, schema_name: str, id_value: str, updated_data: Dict[str, Any]) -> bool:
        """
        Update an existing entry in any CSV file.
        
        Args:
            schema_name: The name of the schema/file
            id_value: The ID value to identify the entry to update
            updated_data: The updated data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if schema exists
        schemas = self._load_schemas()
        if schema_name not in schemas:
            logger.error(f"Schema '{schema_name}' does not exist")
            return False
        
        # Get the ID field name (first required field)
        id_field = schemas[schema_name]["required"][0]
        
        # Load existing data
        data = self._load_csv_data(schema_name)
        
        # Find the entry to update
        for i, entry in enumerate(data):
            if entry.get(id_field) == id_value:
                # Special handling for timetable entries
                if schema_name == "default" or schema_name == "timetable":
                    # Update version if needed
                    if "version" in schemas[schema_name]["attributes"]:
                        current_version = int(entry.get("version", "0"))
                        updated_data["version"] = str(current_version + 1)
                    
                    # Update timestamp if needed
                    if "timestamp" in schemas[schema_name]["attributes"]:
                        updated_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Preserve ID field
                updated_data[id_field] = id_value
                
                # Update the entry
                data[i] = {**entry, **updated_data}
                
                # Save updated data
                self._save_csv_data(schema_name, data)
                logger.info(f"Updated entry {id_value} in '{schema_name}'")
                
                return True
        
        logger.error(f"Entry with {id_field}={id_value} not found in '{schema_name}'")
        return False
    
    def delete_entry(self, schema_name: str, id_value: str) -> bool:
        """
        Delete an entry from any CSV file.
        
        Args:
            schema_name: The name of the schema/file
            id_value: The ID value to identify the entry to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if schema exists
        schemas = self._load_schemas()
        if schema_name not in schemas:
            logger.error(f"Schema '{schema_name}' does not exist")
            return False
        
        # Get the ID field name (first required field)
        id_field = schemas[schema_name]["required"][0]
        
        # Load existing data
        data = self._load_csv_data(schema_name)
        
        # Find and remove the entry
        initial_length = len(data)
        data = [entry for entry in data if entry.get(id_field) != id_value]
        
        if len(data) == initial_length:
            logger.error(f"Entry with {id_field}={id_value} not found in '{schema_name}'")
            return False
        
        # Save updated data
        self._save_csv_data(schema_name, data)
        logger.info(f"Deleted entry {id_value} from '{schema_name}'")
        
        return True
    
    def get_entries(self, schema_name: str) -> List[Dict[str, Any]]:
        """Get all entries for a specific schema/file."""
        return self._load_csv_data(schema_name)
    
    def _get_data_from_agent(self, query: str) -> str:
        """
        Query the TimetableAgent for data.
        
        Args:
            query: The query string to send to the agent
            
        Returns:
            str: The response from the agent
        """
        try:
            return self.query_agent.query(query)
        except Exception as e:
            logger.error(f"Error querying TimetableAgent: {str(e)}")
            return f"Error: {str(e)}"
            
    def create_complex_timetable(self) -> bool:
        """
        Create a complex timetable using data from various sources.
        Leverages the TimetableAgent for querying data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Query for available courses
            courses_response = self._get_data_from_agent("List all courses with their details")
            logger.info(f"Got courses response: {courses_response[:100]}...")
            
            # Query for available staff and their schedules
            staff_response = self._get_data_from_agent("List all staff members and their availability")
            logger.info(f"Got staff response: {staff_response[:100]}...")
            
            # Query for available rooms
            rooms_response = self._get_data_from_agent("List all available rooms")
            logger.info(f"Got rooms response: {rooms_response[:100]}...")
            
            # Now we'll create a timetable entry for each course
            courses_data = self._load_csv_data("courses")
            rooms_data = self._load_csv_data("rooms")
            staff_data = self._load_csv_data("staff_schedules")
            
            # Get existing timetable data
            timetable_data = self._load_csv_data("timetable")
            
            # Create a new timetable entry for each course
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            time_slots = ["09:00-10:30", "10:45-12:15", "13:15-14:45", "15:00-16:30"]
            
            new_entries = []
            created_count = 0
            
            for i, course in enumerate(courses_data):
                course_id = course.get("course_id")
                instructor = course.get("instructor")
                
                # Simple scheduling algorithm
                day_index = i % len(days)
                time_slot_index = i % len(time_slots)
                
                day = days[day_index]
                time_slot = time_slots[time_slot_index]
                
                # Skip lunch time (12:00-13:00)
                if "12:" in time_slot:
                    continue
                
                # Find an available room
                room_id = "1"  # Default to first room
                if rooms_data:
                    room_id = rooms_data[i % len(rooms_data)].get("room_id", "1")
                
                # Create a new timetable entry
                new_entry = {
                    "course": course_id,
                    "room": room_id,
                    "time": time_slot,
                    "day": day,
                    "instructor": instructor,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "created_by": "TimetableCreatorAgent"
                }
                
                # Auto-increment timetable_id
                max_id = 0
                for entry in timetable_data:
                    if entry.get("timetable_id", "").isdigit():
                        max_id = max(max_id, int(entry.get("timetable_id")))
                
                new_entry["timetable_id"] = str(max_id + 1 + len(new_entries))
                new_entry["version"] = "1"
                
                new_entries.append(new_entry)
                created_count += 1
            
            # Add entries to timetable
            timetable_data.extend(new_entries)
            self._save_csv_data("timetable", timetable_data)
            
            logger.info(f"Created {created_count} new timetable entries")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating complex timetable: {str(e)}")
            return False
    
    def process_user_request(self, user_input: str) -> str:
        """
        Process a natural language request from the user and take appropriate action.
        
        Args:
            user_input: The user's request as a natural language string
            
        Returns:
            str: Response to the user
        """
        # If the request is specifically for creating a complex timetable
        if re.search(r'\b(create|generate|make)\b.+\b(complex|integrated|full)\b.+\b(timetable|schedule)\b', user_input, re.IGNORECASE) or \
           re.search(r'\b(timetable|schedule).+\b(based on|using)\b.+\b(courses|staff|rooms)\b', user_input, re.IGNORECASE):
            success = self.create_complex_timetable()
            if success:
                result = "Successfully created a complex timetable based on available courses, staff schedules, and rooms."
                # Save assistant response to memory
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(result)
                return result
            else:
                result = "Failed to create a complex timetable. Please check the logs for details."
                # Save assistant response to memory
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(result)
                return result
        
        # For regular data queries, use the TimetableAgent
        if re.search(r'\b(query|search|find|get|retrieve|show|tell me about)\b', user_input, re.IGNORECASE):
            response = self._get_data_from_agent(user_input)
            # Save the interaction to memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(response)
            return response
            
        # Create a context string with available schemas and commands
        schemas = self.get_available_schemas()
        schema_info = "\n".join([
            f"- {name}: {schema['attributes']}" 
            for name, schema in schemas.items()
        ])
        
        # Build the prompt template
        system_template = """
        You are an AI assistant that helps users create and manage personalized timetables and other related data.
        You can define new schemas, create entries, update entries, and delete entries in various CSV files.
        
        Available schemas (CSV files):
        {schema_info}
        
        You should analyze the user's request and determine what action to take:
        1. Define a new schema (with attributes and required fields)
        2. Create a new entry in a CSV file
        3. Update an existing entry
        4. Delete an entry
        5. List available schemas or entries
        
        IMPORTANT: Pay special attention to which CSV file the user wants to modify. 
        For example, if they want to add a course to courses.csv, use the "courses" schema, not "default" or "timetable".
        
        IMPORTANT: Do NOT use JSON format in your response. Use a simple line-by-line key:value format instead.
        For example:
        
        ACTION: create_entry
        SCHEMA: courses
        course_id: CS105
        course_name: Programming in Python
        instructor: Dr. Smith
        department: Computer Science
        
        Also, always use the current date and time for timestamps, not sample dates.
        
        Respond conversationally to the user, explaining what you're doing.
        """
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Parse the user input using the LLM
        chain = prompt | self.llm | StrOutputParser()
        
        # Get chat history from memory
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Save current input to memory
        self.memory.chat_memory.add_user_message(user_input)
        
        # Invoke chain with history and user input
        response = chain.invoke({
            "schema_info": schema_info,
            "chat_history": chat_history,
            "input": user_input
        })
        
        # Extract action and data from response
        action_match = re.search(r"ACTION:\s*(define_schema|create_entry|update_entry|delete_entry|list_schemas|list_entries)", response)
        schema_match = re.search(r"SCHEMA:\s*(\w+)", response)
        
        if action_match:
            action = action_match.group(1)
            schema_name = schema_match.group(1) if schema_match else None
            
            # Extract data using a more flexible key-value approach instead of JSON
            data = {}
            if schema_name and schema_name in self.schemas:
                # Get all expected attribute names
                attribute_names = self.schemas[schema_name]["attributes"]
                
                # Look for key:value or key=value patterns in the response
                for attr in attribute_names:
                    # Try different patterns for key-value extraction
                    patterns = [
                        fr'{attr}\s*[:=]\s*"([^"]+)"',  # key: "value"
                        fr'{attr}\s*[:=]\s*([^,\n]+)',  # key: value or key=value
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, response)
                        if matches:
                            data[attr] = matches[0].strip()
                            break
            
            try:                
                # Execute the appropriate action
                if action == "define_schema":
                    # Extract attributes and required fields
                    attributes_match = re.search(r"attributes\s*[:=]\s*\[([^\]]+)\]", response)
                    required_match = re.search(r"required\s*[:=]\s*\[([^\]]+)\]", response)
                    
                    if attributes_match:
                        attributes = [attr.strip().strip('"\'') for attr in attributes_match.group(1).split(',')]
                        required = []
                        if required_match:
                            required = [req.strip().strip('"\'') for req in required_match.group(1).split(',')]
                        
                        self.define_schema(schema_name, attributes, required)
                        result = f"Schema '{schema_name}' has been defined with attributes: {attributes}"
                    else:
                        result = "Could not parse attributes for schema definition"
                
                elif action == "create_entry":
                    # Add timestamp using current date and time if needed
                    if schema_name in self.schemas and "timestamp" in self.schemas[schema_name]["attributes"] and "timestamp" not in data:
                        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if self.create_entry(schema_name, data):
                        result = f"Created new entry in '{schema_name}'"
                    else:
                        result = f"Failed to create entry in '{schema_name}'"
                
                elif action == "update_entry":
                    # Get the ID field name (first required field)
                    id_field = self.schemas[schema_name]["required"][0]
                    id_value = data.pop(id_field, None)
                    
                    # Add timestamp using current date and time if needed
                    if "timestamp" in self.schemas[schema_name]["attributes"] and "timestamp" not in data:
                        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if id_value and self.update_entry(schema_name, id_value, data):
                        result = f"Updated entry {id_value} in '{schema_name}'"
                    else:
                        result = f"Failed to update entry in '{schema_name}'"
                
                elif action == "delete_entry":
                    # Get the ID field name (first required field)
                    id_field = self.schemas[schema_name]["required"][0]
                    id_value = data.get(id_field)
                    
                    if id_value and self.delete_entry(schema_name, id_value):
                        result = f"Deleted entry {id_value} from '{schema_name}'"
                    else:
                        result = f"Failed to delete entry in '{schema_name}'"
                
                elif action == "list_schemas":
                    result = f"Available schemas: {list(schemas.keys())}"
                
                elif action == "list_entries":
                    entries = self.get_entries(schema_name)
                    result = f"Found {len(entries)} entries in '{schema_name}'"
                    
                    # Only show first 5 entries to avoid overwhelming response
                    if entries:
                        result += "\nFirst entries:\n"
                        for entry in entries[:5]:
                            result += f"- {entry}\n"
                
                else:
                    result = "Unknown action"
            
            except Exception as e:
                result = f"Error processing action: {str(e)}"
                logger.error(f"Error processing action: {str(e)}")
            
            # Extract the conversational response (everything before ACTION:)
            conversational_response = response.split("ACTION:")[0].strip()
            
            # Construct final response
            final_response = f"{conversational_response}\n\n{result}"
            
            # Save assistant response to memory
            self.memory.chat_memory.add_ai_message(final_response)
            
            return final_response
        
        # If no action was extracted, just return and save the LLM response
        self.memory.chat_memory.add_ai_message(response)
        return response

    def interactive_cli(self):
        """Run an interactive CLI session."""
        print("Welcome to the Timetable Creator Agent!")
        print("Type 'exit' to quit, 'help' for available commands.")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- create schema: Define a new timetable schema")
                print("- list schemas: Show available schemas")
                print("- create entry: Add a new entry to a timetable or other CSV file")
                print("- update entry: Modify an existing entry")
                print("- delete entry: Remove an entry from a timetable or other CSV file")
                print("- list entries: View entries in a timetable or other CSV file")
                print("Or just ask me to help you with any data management task in natural language!")
                continue
            
            # Process the request
            response = self.process_user_request(user_input)
            print(f"\nAgent: {response}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TimetableCreatorAgent CLI')
    parser.add_argument('--model', default=DEFAULT_GROQ_MODEL, 
                       help=f'Groq model to use (default: {DEFAULT_GROQ_MODEL})')
    parser.add_argument('--api-key', help='Groq API key (or set GROQ_API_KEY environment variable)')
    parser.add_argument('--query', help='Run a one-time query')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create the agent
    agent = TimetableCreatorAgent(groq_model=args.model, groq_api_key=args.api_key)
    
    # Process a one-time query if provided
    if args.query:
        response = agent.process_user_request(args.query)
        print(response)
    # Otherwise run in interactive mode
    elif args.interactive or not args.query:
        agent.interactive_cli()