import subprocess
import sys
import os
import tempfile
import ast
from typing import Dict, Any, List, Optional, Annotated
import logging
from dataclasses import dataclass

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # or your preferred LLM



# Define the agent state
@dataclass
class AgentState:
    """State of the AI agent."""
    messages: Annotated[List, add_messages]
    task_description: str = ""
    generated_code: str = ""
    execution_result: Dict[str, Any] = None
    safety_approved: bool = False


class PythonCodeAgent:
    """LangGraph-based AI agent for writing and executing Python code."""
    
    def __init__(self, llm_provider: str = "anthropic", restricted_mode: bool = True):
        """
        Initialize the agent.
        
        Args:
            llm_provider: "openai" or "anthropic"
            restricted_mode: If True, restricts dangerous operations
        """
        self.restricted_mode = restricted_mode
        self.execution_history = []
        self.setup_logging()
        
        # Initialize LLM
        if llm_provider == "openai":
            self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)

        else:
            raise ValueError("Unsupported LLM provider")
        
        # Restricted items for safety
        self.restricted_imports = {
            'os', 'subprocess', 'sys', 'shutil', 'glob', 'pathlib',
            'socket', 'urllib', 'requests', 'http', 'ftplib',
            'smtplib', 'imaplib', 'poplib', 'telnetlib', 'threading',
            'multiprocessing', 'ctypes', 'pickle', 'marshal',
            'importlib'
        }
        
        # Create the agent graph
        self.graph = self._create_agent_graph()
        
    def setup_logging(self):
        """Setup logging for the agent."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    @tool
    def write_python_code(self, task_description: str, requirements: List[str] = None) -> str:
        """
        Generate Python code based on task description.
        
        Args:
            task_description: What the code should accomplish
            requirements: Additional requirements or constraints
            
        Returns:
            Generated Python code
        """
        # Create a detailed prompt for code generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python programmer. Generate clean, well-commented Python code based on the user's requirements.
            
            Guidelines:
            - Write production-quality code with proper error handling
            - Include clear comments explaining the logic
            - Use appropriate libraries and best practices
            - Make the code self-contained and executable
            - Follow PEP 8 style guidelines
            
            Safety restrictions (important):
            - Do not use file system operations (open, read, write files)
            - Do not use network operations (requests, urllib, etc.)
            - Do not use system commands (os.system, subprocess)
            - Do not use dangerous imports like os, sys, subprocess
            - Focus on computational tasks, data processing, and analysis
            """),
            ("human", "Task: {task_description}\nRequirements: {requirements}")
        ])
        
        chain = prompt | self.llm
        result = chain.invoke({
            "task_description": task_description,
            "requirements": requirements or []
        })
        
        # Extract code from the response
        code = result.content
        
        # Clean up the code (remove markdown formatting if present)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    @tool
    def check_code_safety(self, code: str) -> Dict[str, Any]:
        """
        Check if the code is safe to execute.
        
        Args:
            code: Python code to check
            
        Returns:
            Dictionary with safety check results
        """
        if not self.restricted_mode:
            return {"safe": True, "reason": "Unrestricted mode"}
        
        try:
            # Parse the code to check for restricted operations
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for restricted imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.restricted_imports:
                            return {"safe": False, "reason": f"Restricted import: {alias.name}"}
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.restricted_imports:
                        return {"safe": False, "reason": f"Restricted import: {node.module}"}
                
                # Check for file operations
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        return {"safe": False, "reason": "File operations not allowed"}
            
            return {"safe": True, "reason": "Code appears safe"}
            
        except SyntaxError as e:
            return {"safe": False, "reason": f"Syntax error: {e}"}
        except Exception as e:
            return {"safe": False, "reason": f"Error analyzing code: {e}"}
    
    @tool
    def execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name
        
        try:
            import time
            start_time = time.time()
            
            # Execute the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            execution_record = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': execution_time,
                'return_code': result.returncode
            }
            
            self.execution_history.append({
                'code': code,
                'result': execution_record,
                'timestamp': time.time()
            })
            
            return execution_record
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Script execution timed out after {timeout} seconds",
                'output': '',
                'execution_time': timeout,
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'output': '',
                'execution_time': 0,
                'return_code': -1
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(script_path)
            except:
                pass
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph agent workflow."""
        
        # Define tools
        tools = [self.write_python_code, self.check_code_safety, self.execute_python_code]
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Define agent nodes
        def agent_node(state: AgentState) -> AgentState:
            """Main agent reasoning node."""
            messages = state.messages
            
            # Create system message for the agent
            system_msg = SystemMessage(content="""
            You are a helpful AI assistant that can write and execute Python code.
            
            Your workflow:
            1. When given a task, first understand what the user wants
            2. Use the write_python_code tool to generate appropriate code
            3. Use the check_code_safety tool to verify the code is safe
            4. If safe, use the execute_python_code tool to run the code
            5. Report the results back to the user
            
            Always prioritize safety and explain what you're doing at each step.
            """)
            
            # Add system message if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [system_msg] + messages
            
            # Get response from LLM
            response = llm_with_tools.invoke(messages)
            
            return {"messages": [response]}
        
        # Create tool node
        tool_node = ToolNode(tools)
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        # Add memory
        memory = MemorySaver()
        
        return workflow.compile(checkpointer=memory)
    
    def run(self, task: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent on a given task.
        
        Args:
            task: The task description
            thread_id: Thread ID for conversation persistence
            
        Returns:
            Dictionary with the agent's response and execution results
        """
        self.logger.info(f"Running agent on task: {task}")
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=task)],
            task_description=task
        )
        
        # Configure the run
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Run the agent
        result = self.graph.invoke(initial_state, config)
        
        return {
            "messages": result["messages"],
            "final_response": result["messages"][-1].content if result["messages"] else "No response",
            "execution_history": self.execution_history
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        return self.execution_history
    
    def clear_history(self):
        """Clear the execution history."""
        self.execution_history = []


# Example usage and testing
if __name__ == "__main__":
    # Initialize the agent
    print("=== LangGraph Python Code Agent ===")
    print("Initializing agent...")
    
    try:

        
        agent = PythonCodeAgent(llm_provider="openai", restricted_mode=True)
        
        # Example tasks
        tasks = [
            "Create a Python script that calculates the factorial of numbers from 1 to 10",
            "Write a script that generates a simple multiplication table",
            "Create a data analysis script that works with a list of numbers and shows statistics"
        ]
        
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*50}")
            print(f"Task {i}: {task}")
            print(f"{'='*50}")
            
            try:
                result = agent.run(task, thread_id=f"session_{i}")
                print(f"Agent Response: {result['final_response']}")
                
                # Show execution history
                history = agent.get_execution_history()
                if history:
                    latest = history[-1]
                    print(f"\nExecution Result:")
                    print(f"Success: {latest['result']['success']}")
                    if latest['result']['success']:
                        print(f"Output: {latest['result']['output']}")
                    else:
                        print(f"Error: {latest['result']['error']}")
                
            except Exception as e:
                print(f"Error running task: {e}")
                print("Make sure you have set the appropriate API keys as environment variables")
        
        print(f"\nTotal executions: {len(agent.get_execution_history())}")
        
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("Please make sure you have installed the required packages:")
        print("pip install langgraph langchain-openai langchain-anthropic")
        print("And set the appropriate API keys as environment variables")