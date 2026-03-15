import os
import sqlite3

from dotenv import load_dotenv
load_dotenv()
import json
from typing import TypedDict, Annotated, List, Dict, Optional
from datetime import datetime
import operator


from langgraph.graph import StateGraph, END

from google import genai

class DatabaseManager:
    """
    manage the database for the gemini database chatbot
    """
    def __init__(self, db_path: str = "chatbot_database.db"):
        """
         create database connection
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        self.cursor = self.conn.cursor()

        self.setup_database()
    
    def setup_database(self):
        """
        setup the database with the necessary tables
        """

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary REAL,
                joining_date TEXT,
                city TEXT
            )
        """)

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category TEXT,
                price REAL,
                stock INTEGER,
                supplier TEXT
            )
        """)
        
        # Sales table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                product_id INTEGER,
                customer_name TEXT,
                quantity INTEGER,
                sale_date TEXT,
                total_amount REAL,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)

        # Customers table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                city TEXT,
                total_purchases REAL
            )
        """)

        self.cursor.execute("SELECT COUNT(*) FROM employees")
        if self.cursor.fetchone()[0] == 0:
            self._insert_sample_data()
        
        self.conn.commit()
    
    def _insert_sample_data(self):
        """Insert sample data"""
        
        # Employees
        employees = [
            (1, "Rahul Sharma", "Sales", 45000, "2022-01-15", "Mumbai"),
            (2, "Priya Singh", "HR", 38000, "2021-06-20", "Delhi"),
            (3, "Amit Kumar", "IT", 55000, "2020-03-10", "Bangalore"),
            (4, "Sneha Patel", "Sales", 42000, "2022-08-05", "Mumbai"),
            (5, "Vikram Reddy", "IT", 60000, "2019-11-12", "Hyderabad"),
            (6, "Anjali Gupta", "Marketing", 48000, "2021-09-18", "Delhi"),
            (7, "Rohan Mehta", "Finance", 52000, "2020-07-22", "Mumbai"),
            (8, "Neha Joshi", "HR", 40000, "2023-02-14", "Pune"),
        ]
        
        self.cursor.executemany(
            "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
            employees
        )
        
        # Products
        products = [
            (1, "Laptop Dell XPS", "Electronics", 75000, 15, "Dell India"),
            (2, "iPhone 15", "Electronics", 79900, 8, "Apple India"),
            (3, "Office Chair", "Furniture", 8500, 25, "Furniture Co"),
            (4, "Wireless Mouse", "Accessories", 1200, 50, "Logitech"),
            (5, "Monitor 27 inch", "Electronics", 18000, 12, "Samsung"),
            (6, "Desk Lamp", "Furniture", 1500, 30, "Lighting Co"),
            (7, "Keyboard Mechanical", "Accessories", 3500, 20, "Corsair"),
            (8, "Headphones Sony", "Electronics", 5500, 18, "Sony India"),
        ]
        
        self.cursor.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
            products
        )
        
        # Sales
        sales = [
            (1, 1, "Acme Corp", 2, "2024-01-15", 150000),
            (2, 2, "Tech Solutions", 1, "2024-01-20", 79900),
            (3, 3, "Office Plus", 5, "2024-02-10", 42500),
            (4, 4, "StartUp Inc", 10, "2024-02-15", 12000),
            (5, 5, "Design Studio", 3, "2024-03-01", 54000),
            (6, 1, "Mega Corp", 1, "2024-03-05", 75000),
            (7, 7, "Gaming Zone", 4, "2024-03-10", 14000),
            (8, 8, "Music Store", 2, "2024-03-12", 11000),
        ]
        
        self.cursor.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?, ?)",
            sales
        )
        
        # Customers
        customers = [
            (1, "Acme Corp", "contact@acme.com", "9876543210", "Mumbai", 150000),
            (2, "Tech Solutions", "info@techsol.com", "9876543211", "Bangalore", 79900),
            (3, "Office Plus", "sales@officeplus.com", "9876543212", "Delhi", 42500),
            (4, "StartUp Inc", "hello@startup.com", "9876543213", "Pune", 12000),
            (5, "Design Studio", "contact@design.com", "9876543214", "Mumbai", 54000),
        ]
        
        self.cursor.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)",
            customers
        )
        
        self.conn.commit()
        print("✅ Sample database created with data!")

    def execute_query(self, query: str) -> List[Dict]:
        """ 
        execute the query and return the results

        returns list of dictionaries
        """

        try:
            self.cursor.execute(query)
            columns = [description[0] for description in self.cursor.description]

            results = []

            for row in self.cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def get_schema_info(self) -> str:
        """
         return info about the database schema for AI
        """

        schema = """
        DATABASE SCHEMA:
 
1. EMPLOYEES Table:
   - id (INTEGER): Employee ID
   - name (TEXT): Employee name
   - department (TEXT): Department (Sales, HR, IT, Marketing, Finance)
   - salary (REAL): Monthly salary in ₹
   - joining_date (TEXT): Date of joining (YYYY-MM-DD)
   - city (TEXT): City (Mumbai, Delhi, Bangalore, etc.)
 
2. PRODUCTS Table:
   - id (INTEGER): Product ID
   - product_name (TEXT): Name of product
   - category (TEXT): Category (Electronics, Furniture, Accessories)
   - price (REAL): Price in ₹
   - stock (INTEGER): Available quantity
   - supplier (TEXT): Supplier name
 
3. SALES Table:
   - id (INTEGER): Sale ID
   - product_id (INTEGER): Reference to products table
   - customer_name (TEXT): Customer who bought
   - quantity (INTEGER): Number of units sold
   - sale_date (TEXT): Date of sale (YYYY-MM-DD)
   - total_amount (REAL): Total sale amount in ₹
 
4. CUSTOMERS Table:
   - id (INTEGER): Customer ID
   - name (TEXT): Customer name
   - email (TEXT): Email address
   - phone (TEXT): Phone number
   - city (TEXT): City
   - total_purchases (REAL): Total amount spent in ₹
"""
        return schema
    
    def close(self):
        """
        close the database connection
        """
        self.conn.close()
    

class ChatbotState(TypedDict):
    """
    state for the chatbot
    """
    user_question: str

    database_schema: str
    sql_query: str
    query_results: List[Dict]
    final_answer: str
    chat_history: Annotated[List[Dict], operator.add]
    error: Optional[str]
    timestamp: str


class GeminiDatabaseChatbot:

    def __init__(self, gemini_api_key: str, db_path: str = "chatbot_database.db"):
        """
        initialize the chatbot
        """
        # Create client with new google-genai SDK
        # Use Vertex model when GOOGLE_GENAI_USE_VERTEXAI; else use standard Gemini API
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes")
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = "gemini-1.5-flash" if use_vertex else "gemini-2.5-flash"
        
        # Database manager
        self.db = DatabaseManager(db_path)
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
        
        print("✅ Gemini Database Chatbot ready!")
    
    def _build_graph(self) -> StateGraph:
        """
         build LangGraph workflow for the chatbot
        """

        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node("understand_question", self.understand_question)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_query", self.execute_database_query)
        workflow.add_node("format_answer", self.format_final_answer)
        
        # Define flow
        workflow.set_entry_point("understand_question")
        workflow.add_edge("understand_question", "generate_sql")
        workflow.add_edge("generate_sql", "execute_query")
        workflow.add_edge("execute_query", "format_answer")
        workflow.add_edge("format_answer", END)
        
        return workflow.compile()
    

    # ========================================================================
    # NODE 1: UNDERSTAND QUESTION
    # ========================================================================
    
    def understand_question(self, state: ChatbotState) -> ChatbotState:
        """
        Understand the user's question.
        
        What it does:
        - Analyzes the question
        - Adds schema info
        - Sets context
        """
        print("🤔 Understanding your question...")
        
        # Add schema info
        state["database_schema"] = self.db.get_schema_info()
        state["timestamp"] = datetime.now().isoformat()
        
        # Add to chat history
        state["chat_history"] = [{
            "role": "user",
            "content": state["user_question"],
            "timestamp": state["timestamp"]
        }]
        
        print("✅ Question understood!")
        return state
    
    # ========================================================================
    # NODE 2: GENERATE SQL
    # ========================================================================
    
    def generate_sql(self, state: ChatbotState) -> ChatbotState:
        """
        Generate SQL query from natural language question.
        
        Uses Gemini AI to generate SQL.
        """
        print("🔧 Generating SQL query...")
        
        prompt = f"""
You are an expert SQL developer. Convert the natural language question to SQL query.
 
DATABASE SCHEMA:
{state['database_schema']}
 
USER QUESTION: {state['user_question']}
 
INSTRUCTIONS:
1. Generate ONLY the SQL query, nothing else
2. Use proper SQL syntax for SQLite
3. Make sure the query is safe (no DROP, DELETE, UPDATE)
4. Use appropriate JOINs if needed
5. Return ONLY the SQL query without any markdown, explanations, or formatting
 
EXAMPLES:
Question: "How many employees are there?"
SQL: SELECT COUNT(*) as total_employees FROM employees
 
Question: "Show me all products under 10000"
SQL: SELECT * FROM products WHERE price < 10000
 
Question: "Who are the top 3 customers by spending?"
SQL: SELECT name, total_purchases FROM customers ORDER BY total_purchases DESC LIMIT 3
 
Now generate SQL for the user's question:
"""
        
        try:
            # Generate SQL from Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            sql_query = response.text.strip()
            
            # Clean up SQL (remove markdown if any)
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            state["sql_query"] = sql_query
            state["error"] = None
            
            print(f"✅ SQL generated: {sql_query}")
            
        except Exception as e:
            state["error"] = f"SQL generation failed: {str(e)}"
            print(f"❌ Error: {state['error']}")
        
        return state
    
    # ========================================================================
    # NODE 3: EXECUTE QUERY
    # ========================================================================
    
    def execute_database_query(self, state: ChatbotState) -> ChatbotState:
        """
        Run the generated SQL on the database.
        """
        print("💾 Executing database query...")
        
        if state.get("error"):
            return state
        
        try:
            # Safety check - read-only queries
            sql_lower = state["sql_query"].lower()
            dangerous_keywords = ["drop", "delete", "update", "insert", "alter", "create"]
            
            if any(keyword in sql_lower for keyword in dangerous_keywords):
                state["error"] = "Security: Only SELECT queries are allowed"
                return state
            
            # Execute query
            results = self.db.execute_query(state["sql_query"])
            state["query_results"] = results
            
            print(f"✅ Query executed! Found {len(results)} results")
            
        except Exception as e:
            state["error"] = f"Query execution failed: {str(e)}"
            print(f"❌ Error: {state['error']}")
        
        return state
    
    # ========================================================================
    # NODE 4: FORMAT ANSWER
    # ========================================================================
    
    def format_final_answer(self, state: ChatbotState) -> ChatbotState:
        """
        Convert results into a user-friendly answer.
        
        Uses Gemini AI to generate a natural language answer.
        """
        print("💬 Formatting answer...")
        
        if state.get("error"):
            state["final_answer"] = f"❌ Error: {state['error']}"
            return state
        
        # If no results
        if not state["query_results"]:
            state["final_answer"] = "No data found for your question. Try asking something else!"
            return state
        
        # Format answer using Gemini
        prompt = f"""
You are a helpful assistant. Convert the database query results into a natural, conversational answer in English.
 
USER QUESTION: {state['user_question']}
 
SQL QUERY USED: {state['sql_query']}
 
QUERY RESULTS:
{json.dumps(state['query_results'], indent=2)}
 
INSTRUCTIONS:
1. Give a clear, concise answer in English
2. Use emojis to make it friendly
3. If there are numbers, format them nicely (use ₹ for money, add commas)
4. If results are in a table format, present them clearly
5. Keep it conversational and helpful
6. Don't mention the SQL query unless asked
 
EXAMPLES:
Question: "How many employees are there?"
Answer: "We have a total of **8 employees**! 👥"
 
Question: "Top 3 expensive products?"
Answer: "Here are the top 3 most expensive products:
1. 📱 iPhone 15 - ₹79,900
2. 💻 Laptop Dell XPS - ₹75,000
3. 🖥️ Monitor 27 inch - ₹18,000"
 
Now format the answer:
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            state["final_answer"] = response.text.strip()
            
            # Add to chat history
            state["chat_history"] = [{
                "role": "assistant",
                "content": state["final_answer"],
                "sql_used": state["sql_query"],
                "timestamp": datetime.now().isoformat()
            }]
            
            print("✅ Answer ready!")
            
        except Exception as e:
            state["final_answer"] = f"Answer formatting failed: {str(e)}"
            print(f"❌ Error: {state['final_answer']}")
        
        return state
    
    # ========================================================================
    # MAIN CHATBOT FUNCTION
    # ========================================================================
    
    def chat(self, question: str) -> str:
        """
        Main chat function - use this to ask questions about the database.
        
        Args:
            question: The user's question
            
        Returns:
            Answer as string
        """
        
        # Initial state
        initial_state = ChatbotState(
            user_question=question,
            database_schema="",
            sql_query="",
            query_results=[],
            final_answer="",
            chat_history=[],
            error=None,
            timestamp=""
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state["final_answer"]
    
    def close(self):
        """Cleanup"""
        self.db.close()

# ============================================================================
# DEMO / EXAMPLE USAGE
# ============================================================================
 
def demo_chatbot():
    """
    Demo function - use the chatbot like this.
    """
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║      🤖 GEMINI DATABASE CHATBOT 🤖                             ║
    ║                                                                ║
    ║  FREE Gemini API + LangGraph + SQLite Database                ║
    ║  Cost: ₹0 (Completely FREE!)                                  ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Get API key (from environment variable or direct)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "your-gemini-api-key-here"
    
    if api_key == "your-gemini-api-key-here":
        print("""
⚠️  GEMINI API KEY SETUP:
 
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google account (FREE!)
3. Create API key
4. Copy and use below:
 
export GEMINI_API_KEY='your-actual-key'
 
Or edit this file and paste key directly.
        """)
        return
    
    # Create chatbot
    bot = GeminiDatabaseChatbot(gemini_api_key=api_key)
    
    print("\n" + "="*70)
    print("CHATBOT READY! Ask questions about the database")
    print("="*70 + "\n")
    
    # Sample questions
    sample_questions = [
        "How many employees are there in total?",
        "What is the most expensive product?",
        "How many people are in the IT department?",
        "How many employees are in Mumbai?",
        "Show top 3 customers by spending",
        "Which products are under 5000 rupees?",
        "What is the total sales amount?",
        "What is the average employee salary?"
    ]
    
    print("📝 SAMPLE QUESTIONS (Try these!):\n")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("(Type 'exit' to quit, 'samples' to see questions again)\n")
    
    # Interactive mode
    while True:
        try:
            question = input("\n💬 Your question: ").strip()
            
            if question.lower() == 'exit':
                print("\n👋 Bye! Shutting down database chatbot...")
                break
            
            if question.lower() == 'samples':
                print("\n📝 SAMPLE QUESTIONS:\n")
                for i, q in enumerate(sample_questions, 1):
                    print(f"{i}. {q}")
                continue
            
            if not question:
                continue
            
            print("\n" + "-"*70)
            
            # Get answer
            answer = bot.chat(question)
            
            print(f"\n🤖 Answer:\n{answer}")
            print("-"*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted! Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Cleanup
    bot.close()
    print("\n✅ Chatbot closed successfully!")
 
 
if __name__ == "__main__":
    demo_chatbot()