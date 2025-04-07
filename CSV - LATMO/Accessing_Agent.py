"""
LangChain Agent for accessing and querying CSV data from database directory
Uses Ollama models for LLM and embeddings
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.messages import HumanMessage, AIMessage
import logging
import argparse

import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
TIMETABLES_PATH = os.path.join(DATABASE_PATH, "timetables")
DEFAULT_LLM_MODEL = "gemma3:4b"  # Default model
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # This model supports embeddings in Ollama

class TimetableAgent:
    """Agent for querying timetable data from CSV files using LangChain and Ollama."""
    
    def __init__(self, llm_model: str = DEFAULT_LLM_MODEL, embedding_model: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the agent with the specified Ollama model."""
        logger.info(f"Initializing TimetableAgent with LLM model: {llm_model} and embedding model: {embedding_model}")
        
        try:
            # Using ChatOllama as it's more compatible with Ollama
            self.llm = ChatOllama(model=llm_model, temperature=0.1)
            self.embeddings = OllamaEmbeddings(model=embedding_model)
            self.csv_files = self._get_csv_files()
            self.retriever = self._setup_vectorstore()
            self.chain = self._create_chain()
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise
    
    def _get_csv_files(self) -> List[str]:
        """Get all CSV files from the timetables directory."""
        logger.info(f"Searching for CSV files in {TIMETABLES_PATH}")
        files = [
            os.path.join(TIMETABLES_PATH, f) 
            for f in os.listdir(TIMETABLES_PATH) 
            if f.endswith('.csv')
        ]
        logger.info(f"Found {len(files)} CSV files")
        return files
    
    def _setup_vectorstore(self):
        """Load CSV files and create a vectorstore retriever."""
        logger.info("Setting up vector store from CSV files")
        documents = []
        
        for csv_file in self.csv_files:
            # Get the filename without extension as a source identifier
            source_name = os.path.basename(csv_file).replace('.csv', '')
            logger.info(f"Processing {source_name} CSV file")
            
            try:
                loader = CSVLoader(
                    file_path=csv_file,
                    source_column=None,
                    metadata_columns=[],
                    csv_args={
                        'delimiter': ',',
                        'quotechar': '"',
                    }
                )
                
                # Add file name as metadata for better context
                file_docs = loader.load()
                logger.info(f"Loaded {len(file_docs)} documents from {source_name}")
                
                for doc in file_docs:
                    # Add source filename to metadata
                    doc.metadata["source"] = os.path.basename(csv_file)
                    # Add content type
                    doc.metadata["content_type"] = "timetable_data"
                
                documents.extend(file_docs)
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {str(e)}")
                raise
        
        logger.info(f"Creating vector store with {len(documents)} total documents")
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=self.embeddings,
            persist_directory=os.path.join(DATABASE_PATH, "chroma_db")
        )
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    def _create_chain(self):
        """Create the retrieval chain with the prompt template."""
        logger.info("Creating retrieval chain with prompt template")
        
        # Using a simpler RetrievalQA chain which handles context properly
        template = """
        You are an assistant that helps answer questions about timetable data. 
        
        Be accurate, helpful, and base your answers solely on the provided CSV data.
        If the information is not in the provided data, admit that you don't know.
        
        Available data sources:
        - courses.csv: Information about courses (course_id, course_name, instructor, department)
        - enrollments.csv: Student enrollments in courses
        - rooms.csv: Available rooms
        - staff_schedules.csv: Staff availability and schedules
        - students.csv: Student information
        - timetable.csv: Course scheduling information
        
        When referring to specific data, cite which file it came from.
        
        Question: {question}
        
        {context}
        
        Answer:
        """
        
        # Create a simpler RetrievalQA chain which is more reliable
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate.from_template(template),
            },
            return_source_documents=True,
        )
        
        return qa_chain
    
    def query(self, question: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """Process a question and return an answer based on the CSV data."""
        logger.info(f"Processing query: {question}")
        if chat_history is None:
            chat_history = []
        
        try:    
            # Simple query with RetrievalQA
            result = self.chain({"query": question})
            
            # Get the answer and source documents
            answer = result.get('result', '')
            source_docs = result.get('source_documents', [])
            
            # Log sources for debugging
            sources = set()
            for doc in source_docs:
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
            
            if sources:
                logger.info(f"Retrieved data from: {', '.join(sources)}")
            
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}"

def list_available_models():
    """List available Ollama models for the user"""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error listing models: {str(e)}"
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TimetableAgent CLI')
    parser.add_argument('--llm', default=DEFAULT_LLM_MODEL, help=f'Ollama LLM model (default: {DEFAULT_LLM_MODEL})')
    parser.add_argument('--embedding', default=DEFAULT_EMBEDDING_MODEL, help=f'Ollama embedding model (default: {DEFAULT_EMBEDDING_MODEL})')
    parser.add_argument('--list-models', action='store_true', help='List available Ollama models')
    parser.add_argument('--query', help='Run a one-time query')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.list_models:
        print("Available Ollama models:")
        print(list_available_models())
        exit(0)
    
    try:
        # Initialize agent
        print(f"Initializing Timetable Agent with LLM: {args.llm}, Embedding: {args.embedding}...")
        agent = TimetableAgent(llm_model=args.llm, embedding_model=args.embedding)
        
        if args.query:
            # One-time query mode
            answer = agent.query(args.query)
            print(f"Question: {args.query}\nAnswer: {answer}")
        elif args.interactive:
            # Interactive mode
            print("\nTimetable Query Assistant")
            print("=========================")
            print("Type 'exit' to quit\n")
            
            chat_history = []
            
            while True:
                query = input("Your question: ")
                
                if query.lower() in ["exit", "quit"]:
                    break
                    
                try:
                    answer = agent.query(query)
                    print(f"\nAnswer: {answer}\n")
                    
                    # Add to chat history (not currently used in new implementation)
                    chat_history.append({"type": "human", "content": query})
                    chat_history.append({"type": "ai", "content": answer})
                except Exception as e:
                    print(f"Error: {str(e)}")
        else:
            # Default mode - run one test query
            question = "How many students are enrolled in total?"
            print(f"Processing query: {question}")
            answer = agent.query(question)
            print(f"Question: {question}\nAnswer: {answer}")
            print("\nRun with --interactive flag for interactive mode")
            print("Run with --list-models to see available Ollama models")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTROUBLESHOOTING:")
        print("1. Make sure Ollama is running (run 'ollama serve' in a separate terminal)")
        print("2. Make sure the required models are pulled. Recommended models:")
        print("   - For LLM: ollama pull gemma3:4b")
        print("   - For embeddings: ollama pull nomic-embed-text")
        print("\nAvailable commands:")
        print("- List models: python agent.py --list-models")
        print("- Specify models: python agent.py --llm gemma3:4b --embedding nomic-embed-text")
        print("- Interactive mode: python agent.py --interactive")
        print("- One-time query: python agent.py --query \"How many students are enrolled?\"")
