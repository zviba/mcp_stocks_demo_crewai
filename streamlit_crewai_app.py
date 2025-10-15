#!/usr/bin/env python3
"""
Streamlit CrewAI + MCP Stocks Analysis Demo

This Streamlit app provides a web interface for running CrewAI stock analysis
using the MCP stocks server. Users can select stocks, configure agents, and
view comprehensive analysis results.
"""

import streamlit as st
import json
import os
import sys
import time
import threading
import subprocess
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Import CrewAI components
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    st.error("CrewAI not installed. Please run: pip install crewai")

# Fallback BaseTool class if CrewAI is not available
if not CREWAI_AVAILABLE:
    class BaseTool:
        def __init__(self, name: str, description: str):
            self.name = name
            self.description = description

# MCP Server Configuration
MCP_SERVER_URL = "http://127.0.0.1:8001"

# Page configuration
st.set_page_config(
    page_title="CrewAI + MCP Stocks Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
        color: #333333;
    }
    .agent-card h4 {
        color: #1f77b4;
        margin-top: 0;
    }
    .agent-card p {
        color: #333333;
        margin: 0.5rem 0;
    }
    .agent-card strong {
        color: #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# MCP Tool Classes (same as in crewai_stocks_demo.py)
class MCPTool(BaseTool):
    """Base class for MCP server tools"""
    
    def __init__(self, endpoint: str, name: str, description: str):
        super().__init__(name=name, description=description)
        # Store endpoint as a class attribute instead of instance attribute
        self._endpoint = endpoint
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool via HTTP request"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json=kwargs,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"
    
    @property
    def endpoint(self):
        """Get the endpoint for this tool"""
        return self._endpoint

class SearchSymbolsTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/search",
            name="search_symbols",
            description="Search for stock symbols by company name or ticker. Requires 'q' parameter."
        )
    
    def _run(self, q: str, **kwargs) -> str:
        """Execute the search tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={"q": q},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class GetQuoteTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/quote",
            name="get_quote",
            description="Get latest price, change percentage, and volume for a stock. Requires symbol parameter."
        )
    
    def _run(self, symbol: str, **kwargs) -> str:
        """Execute the quote tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={"symbol": symbol},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class GetPriceSeriesTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/series",
            name="get_price_series",
            description="Get historical OHLCV price data for a stock. Requires symbol parameter."
        )
    
    def _run(self, symbol: str, **kwargs) -> str:
        """Execute the price series tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={"symbol": symbol},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class GetIndicatorsTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/indicators",
            name="get_indicators",
            description="Get technical indicators (SMA, EMA, RSI) for a stock. Requires symbol parameter."
        )
    
    def _run(self, symbol: str, window_sma: int = 20, window_ema: int = 50, window_rsi: int = 14, **kwargs) -> str:
        """Execute the indicators tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={
                    "symbol": symbol,
                    "window_sma": window_sma,
                    "window_ema": window_ema,
                    "window_rsi": window_rsi
                },
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class GetEventsTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/events",
            name="get_events",
            description="Detect market events like gaps, volatility spikes, and 52-week extremes. Requires symbol parameter."
        )
    
    def _run(self, symbol: str, **kwargs) -> str:
        """Execute the events tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={"symbol": symbol},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class GetExplanationTool(MCPTool):
    def __init__(self):
        super().__init__(
            endpoint="/explain",
            name="get_explanation",
            description="Get AI-powered explanation of technical analysis with market context. Requires symbol and openai_api_key parameters."
        )
    
    def _run(self, symbol: str, openai_api_key: str, language: str = "en", tone: str = "neutral", 
             risk_profile: str = "balanced", horizon_days: int = 30, bullets: bool = True, **kwargs) -> str:
        """Execute the explanation tool with required parameters"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self._endpoint}",
                json={
                    "symbol": symbol,
                    "language": language,
                    "tone": tone,
                    "risk_profile": risk_profile,
                    "horizon_days": horizon_days,
                    "bullets": bullets,
                    "openai_api_key": openai_api_key
                },
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

def check_mcp_server() -> bool:
    """Check if MCP server is running"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_mcp_server():
    """Start MCP server in background"""
    if not check_mcp_server():
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", 
                "api:app", 
                "--host", "127.0.0.1", 
                "--port", "8001",
                "--log-level", "warning"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            for _ in range(20):
                time.sleep(0.5)
                if check_mcp_server():
                    return True
            return False
        except Exception:
            return False
    return True

def create_agents(openai_api_key: str) -> Dict[str, Agent]:
    """Create CrewAI agents"""
    if not CREWAI_AVAILABLE:
        return {}
    
    # Initialize MCP tools
    search_tool = SearchSymbolsTool()
    quote_tool = GetQuoteTool()
    series_tool = GetPriceSeriesTool()
    indicators_tool = GetIndicatorsTool()
    events_tool = GetEventsTool()
    explanation_tool = GetExplanationTool()
    
    # Research Agent
    research_agent = Agent(
        role="Stock Research Specialist",
        goal="Gather comprehensive basic information about stocks including current quotes, historical data, and company details",
        backstory="""You are an experienced stock researcher with deep knowledge of financial markets. 
        Your expertise lies in efficiently gathering and organizing stock data from multiple sources. 
        You excel at finding relevant information quickly and presenting it in a clear, structured format.""",
        tools=[search_tool, quote_tool, series_tool],
        verbose=True,  # Enable verbose output
        allow_delegation=False
    )
    
    # Technical Analyst
    technical_agent = Agent(
        role="Technical Analysis Expert",
        goal="Perform detailed technical analysis using indicators, patterns, and market events to assess stock momentum and trends",
        backstory="""You are a seasoned technical analyst with 15+ years of experience in chart analysis and market indicators. 
        You specialize in interpreting technical signals, identifying patterns, and understanding market psychology. 
        Your analysis is methodical and based on proven technical analysis principles.""",
        tools=[indicators_tool, events_tool, explanation_tool],
        verbose=True,  # Enable verbose output
        allow_delegation=False
    )
    
    # Report Writer
    report_agent = Agent(
        role="Financial Report Writer",
        goal="Create comprehensive, well-structured investment reports that synthesize research and technical analysis into actionable insights",
        backstory="""You are a professional financial writer with expertise in translating complex market data into clear, 
        actionable reports. You have a talent for presenting technical information in an accessible way while maintaining 
        accuracy and professional standards. Your reports are known for their clarity and practical insights.""",
        tools=[],
        verbose=True,  # Enable verbose output
        allow_delegation=False
    )
    
    return {
        "research": research_agent,
        "technical": technical_agent,
        "report": report_agent
    }

def create_tasks(symbol: str, openai_api_key: str) -> List[Task]:
    """Create CrewAI tasks"""
    if not CREWAI_AVAILABLE:
        return []
    
    # Research Task
    research_task = Task(
        description=f"""
        Conduct comprehensive research on the stock symbol '{symbol}'. Your research should include:
        
        1. Search for the symbol to verify it exists and get company information
        2. Get the latest quote including current price, change percentage, and volume
        3. Retrieve historical price data (180 days) to understand recent price movements
        4. Organize all data in a clear, structured format
        
        Focus on accuracy and completeness. Present your findings in a well-organized manner that will be useful for technical analysis.
        """,
        expected_output="A comprehensive research report containing: symbol verification, current quote data, and historical price summary with key statistics",
        agent=None,
        tools=[SearchSymbolsTool(), GetQuoteTool(), GetPriceSeriesTool()]
    )
    
    # Technical Analysis Task
    technical_task = Task(
        description=f"""
        Perform detailed technical analysis on '{symbol}' using the research data provided. Your analysis should include:
        
        1. Calculate and interpret technical indicators (SMA, EMA, RSI)
        2. Identify significant market events (gaps, volatility spikes, 52-week extremes)
        3. Use AI explanation tool to get contextual interpretation of technical signals
        4. Assess overall technical momentum and trend direction
        5. Identify key support and resistance levels from the data
        
        Provide specific, actionable technical insights based on the data analysis.
        """,
        expected_output="A detailed technical analysis report with indicator interpretations, event analysis, trend assessment, and key price levels",
        agent=None,
        tools=[GetIndicatorsTool(), GetEventsTool(), GetExplanationTool()],
        context=[research_task]
    )
    
    # Report Task
    report_task = Task(
        description=f"""
        Create a comprehensive investment analysis report for '{symbol}' that synthesizes all research and technical analysis. 
        The report should include:
        
        1. Executive Summary with key findings
        2. Current Market Position (price, volume, recent changes)
        3. Technical Analysis Summary (indicators, events, trends)
        4. Risk Assessment based on technical signals
        5. Key Takeaways and Observations
        6. Professional disclaimers
        
        Write in a professional, clear style suitable for investment decision-making. 
        Include specific data points and technical observations from the analysis.
        """,
        expected_output="A professional investment analysis report with executive summary, technical analysis, risk assessment, and key takeaways",
        agent=None,
        context=[research_task, technical_task]
    )
    
    return [research_task, technical_task, report_task]

def run_crewai_analysis(symbol: str, openai_api_key: str, progress_callback=None, verbose_callback=None) -> Dict[str, Any]:
    """Run CrewAI analysis with progress tracking"""
    if not CREWAI_AVAILABLE:
        return {"error": "CrewAI not available"}
    
    try:
        # Create agents
        if progress_callback:
            progress_callback("🔧 Creating specialized agents...", 5)
        if verbose_callback:
            verbose_callback("Initializing specialized AI agents for stock analysis...")
        
        agents = create_agents(openai_api_key)
        if verbose_callback:
            verbose_callback(f"Created {len(agents)} specialized agents: Research, Technical Analysis, and Report Writing")
        
        # Create tasks
        if progress_callback:
            progress_callback("📋 Setting up analysis tasks...", 8)
        if verbose_callback:
            verbose_callback("Setting up analysis workflow tasks...")
        
        tasks = create_tasks(symbol, openai_api_key)
        if verbose_callback:
            verbose_callback(f"Created {len(tasks)} analysis tasks for symbol {symbol}")
        
        # Assign agents to tasks
        tasks[0].agent = agents["research"]
        tasks[1].agent = agents["technical"]
        tasks[2].agent = agents["report"]
        if verbose_callback:
            verbose_callback("Assigned agents to their respective tasks")
        
        # Create crew
        if progress_callback:
            progress_callback("👥 Assembling analysis crew...", 10)
        if verbose_callback:
            verbose_callback("Assembling collaborative AI crew for sequential analysis...")
        
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True  # Enable verbose output for the crew
        )
        if verbose_callback:
            verbose_callback("Crew assembled successfully - ready to begin analysis")
        
        # Execute analysis with detailed progress tracking
        if progress_callback:
            progress_callback("🚀 Starting comprehensive analysis...", 10)
        
        # Use Crew's built-in execution instead of individual task execution
        if progress_callback:
            progress_callback("📊 Executing crew workflow...", 20)
        
        # Test MCP server connectivity before starting crew
        if progress_callback:
            progress_callback("🔍 Testing MCP server connectivity...", 15)
        if verbose_callback:
            verbose_callback("Testing MCP server connectivity for data access...")
        
        try:
            test_response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
            if test_response.status_code != 200:
                raise Exception(f"MCP server returned status {test_response.status_code}")
            if verbose_callback:
                verbose_callback("✅ MCP server is responding - data access confirmed")
        except Exception as e:
            if verbose_callback:
                verbose_callback(f"❌ MCP server connectivity failed: {str(e)}")
            raise Exception(f"MCP server is not responding: {str(e)}")
        
        # Set OpenAI API key as environment variable for CrewAI
        import os
        os.environ["OPENAI_API_KEY"] = openai_api_key
        if verbose_callback:
            verbose_callback("OpenAI API key configured for LLM operations")
        
        # Execute the crew workflow with timeout
        if progress_callback:
            progress_callback("📊 Executing crew workflow...", 20)
        if verbose_callback:
            verbose_callback("🚀 Starting crew execution - agents will now begin their analysis...")
            verbose_callback("📋 Task 1: Research Agent will gather stock data...")
            verbose_callback("📊 Task 2: Technical Agent will analyze indicators...")
            verbose_callback("📝 Task 3: Report Agent will compile final analysis...")
        
        # Add timeout protection for crew execution
        import threading
        
        def timeout_handler():
            if verbose_callback:
                verbose_callback("⏰ Crew execution timed out after 2 minutes")
            raise TimeoutError("Crew execution timed out")
        
        # Set up timeout
        timer = threading.Timer(120.0, timeout_handler)  # 2 minutes timeout
        timer.start()
        
        try:
            # Update progress before crew execution
            if progress_callback:
                progress_callback("🔄 Executing crew tasks...", 30)
            if verbose_callback:
                verbose_callback("⚡ Crew execution starting now...")
            
            # Capture CrewAI verbose output
            import sys
            from io import StringIO
            
            # Create a custom stdout/stderr capture
            class VerboseCapture:
                def __init__(self, callback):
                    self.callback = callback
                    self.buffer = StringIO()
                
                def write(self, text):
                    if text.strip():  # Only capture non-empty lines
                        self.callback(f"🤖 {text.strip()}")
                    return len(text)
                
                def flush(self):
                    pass
            
            # Capture stdout during crew execution
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            if verbose_callback:
                verbose_capture = VerboseCapture(verbose_callback)
                sys.stdout = verbose_capture
                sys.stderr = verbose_capture
            
            result = crew.kickoff()
            
            # Restore stdout/stderr
            if verbose_callback:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            timer.cancel()  # Cancel timeout if successful
            if verbose_callback:
                verbose_callback("✅ Crew execution completed successfully - analysis finished")
        except Exception as e:
            # Restore stdout/stderr in case of error
            if verbose_callback:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            timer.cancel()  # Cancel timeout on error
            if verbose_callback:
                verbose_callback(f"❌ Crew execution failed: {str(e)}")
            raise e
        
        # Final completion message
        if progress_callback:
            progress_callback("🎉 Analysis completed successfully!", 100)
        
        return {
            "success": True,
            "result": str(result),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol
        }

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CrewAI + MCP Stocks Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive stock analysis using specialized AI agents**")
    
    # Usage Instructions
    with st.expander("📖 How to Use This Application", expanded=False):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        **Step 1: Start the MCP Server**
        - Check the sidebar for "🔗 MCP Server Status"
        - If it shows "❌ MCP Server Not Running", click the "🚀 Start MCP Server" button
        - Wait for "✅ MCP Server Connected" status
        
        **Step 2: Enter Your OpenAI API Key**
        - In the sidebar, enter your OpenAI API key in the "🔑 OpenAI API Key" field
        - This is required for AI-powered explanations in the analysis
        - You'll see "✅ API Key Set" when successful
        
        **Step 3: Enter Stock Symbol**
        - In the main interface, enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
        - The system will search for the symbol and display company information
        
        **Step 4: Start Analysis**
        - Click the "🔍 Start Analysis" button
        - Watch the progress bar and agent activity log
        - Results will appear in the Analysis Results section
        
        ### 🔧 What Happens During Analysis
        
        **Three AI Agents Work Together:**
        1. **Research Agent** - Gathers stock data, quotes, and historical prices
        2. **Technical Agent** - Analyzes indicators, events, and market patterns  
        3. **Report Agent** - Compiles findings into a comprehensive report
        
        **Data Sources:**
        - Real-time stock quotes and historical data
        - Technical indicators (SMA, EMA, RSI)
        - Market events (gaps, volatility spikes, 52-week extremes)
        - AI-powered explanations and insights
        
        ### 🛠️ Troubleshooting
        
        **MCP Server Issues:**
        - Check the sidebar "🔗 MCP Server Status"
        - If not connected, click "🚀 Start MCP Server" button
        - Wait for "✅ MCP Server Connected" status before proceeding
        
        **API Key Issues:**
        - Verify your OpenAI API key is valid and has credits
        - Check sidebar shows "✅ API Key Set"
        - The key is required for LLM explanations
        
        **Analysis Stuck:**
        - Check the Agent Activity Log for error messages
        - Ensure MCP server is running and API key is set
        - Try a different stock symbol
        - Use "🗑️ Clear Results" button to reset if needed
        
        ### 📊 Understanding the Output
        
        **Analysis Results:**
        - Comprehensive stock analysis report
        - Technical indicators and market insights
        - Downloadable report in text format
        
        **Agent Activity Log:**
        - Real-time view of what each agent is doing
        - Tool usage and data retrieval
        - Error messages and debugging information
        
        ### 🔗 Additional Resources
        
        - **CrewAI Documentation**: https://docs.crewai.com/
        - **MCP Protocol**: https://modelcontextprotocol.io/
        - **OpenAI API**: https://platform.openai.com/
        """)
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # MCP Server Status
        st.subheader("🔗 MCP Server Status")
        if check_mcp_server():
            st.success("✅ MCP Server Connected")
        else:
            st.error("❌ MCP Server Not Running")
            if st.button("🚀 Start MCP Server", use_container_width=True):
                with st.spinner("Starting MCP server..."):
                    if start_mcp_server():
                        st.success("✅ MCP server started successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start MCP server")
        
        st.markdown("---")
        
        # OpenAI API Key
        st.subheader("🔑 OpenAI API Key")
        openai_api_key = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Required for LLM explanations in technical analysis"
        )
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key
            st.success("✅ API Key Set")
        else:
            st.warning("⚠️ API Key Required")
        
        # CrewAI Status
        st.subheader("🤖 CrewAI Status")
        if CREWAI_AVAILABLE:
            st.markdown('<p class="status-success">✅ CrewAI Available</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">❌ CrewAI Not Installed</p>', unsafe_allow_html=True)
            st.code("pip install crewai langchain langchain-openai")
    
    # Main content
    if not CREWAI_AVAILABLE:
        st.error("CrewAI is not installed. Please install it using the command in the sidebar.")
        return
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to enable LLM explanations.")
    
    # Stock selection
    st.header("📊 Stock Selection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input(
            "Stock Symbol",
            value=st.session_state.get("selected_symbol", "AAPL"),
            placeholder="Enter stock symbol (e.g., AAPL, NVDA, TSLA)",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        if symbol:
            st.session_state["selected_symbol"] = symbol
    
    with col2:
        st.metric("Selected Symbol", symbol if symbol else "None")
    
    # Agent information
    st.header("👥 Analysis Agents")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 Research Agent</h4>
            <p><strong>Role:</strong> Stock Research Specialist</p>
            <p><strong>Tools:</strong> Search, Quote, Price Series</p>
            <p><strong>Output:</strong> Basic stock data and historical information</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h4>📈 Technical Analyst</h4>
            <p><strong>Role:</strong> Technical Analysis Expert</p>
            <p><strong>Tools:</strong> Indicators, Events, AI Explanation</p>
            <p><strong>Output:</strong> Technical analysis and market insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-card">
            <h4>📝 Report Writer</h4>
            <p><strong>Role:</strong> Financial Report Writer</p>
            <p><strong>Tools:</strong> Synthesis only</p>
            <p><strong>Output:</strong> Comprehensive investment report</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis controls
    st.header("🚀 Run Analysis")
    
    if not symbol:
        st.warning("Please enter a stock symbol to begin analysis.")
        return
    
    if not check_mcp_server():
        st.error("MCP server is not running. Please start it using the button in the sidebar.")
        return
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        run_analysis = st.button("🔍 Start Analysis", type="primary", use_container_width=True)
    
    with col2:
        clear_results = st.button("🗑️ Clear Results", use_container_width=True)
    
    with col3:
        # Show analysis status
        if st.session_state.get("analysis_running", False):
            st.markdown('<p class="status-warning">⏳ Analysis in progress...</p>', unsafe_allow_html=True)
            
            # Show verbose messages during analysis if they exist
            if "verbose_messages" in st.session_state and st.session_state["verbose_messages"]:
                st.subheader("🤖 Agent Activity Log")
                for msg in st.session_state["verbose_messages"][-20:]:  # Show last 20 messages
                    st.text(msg)
        elif "analysis_result" in st.session_state:
            if st.session_state["analysis_result"].get("success", False):
                st.markdown('<p class="status-success">✅ Analysis completed!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Analysis failed</p>', unsafe_allow_html=True)
    
    
    # Clear results
    if clear_results:
        for key in ["analysis_result", "analysis_running", "analysis_progress", "debug_messages", "verbose_messages"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Run analysis
    if run_analysis and symbol and openai_api_key and not st.session_state.get("analysis_running", False):
        # Set flag to prevent multiple executions
        st.session_state["analysis_running"] = True
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.subheader("📊 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_percentage = st.empty()
        
        # Create Analysis Results section (empty initially)
        st.header("📋 Analysis Results")
        results_placeholder = st.empty()
        
        # Create persistent verbose container outside progress container
        st.subheader("🤖 Agent Activity Log")
        verbose_placeholder = st.empty()
        
        # Progress callback with percentage tracking
        def update_progress(message, percentage=None):
            status_text.text(message)
            if percentage is not None:
                progress_bar.progress(percentage)
                progress_percentage.text(f"Progress: {percentage:.0f}%")
        
        # Verbose callback to capture agent activities
        def verbose_callback(message):
            if "verbose_messages" not in st.session_state:
                st.session_state["verbose_messages"] = []
            st.session_state["verbose_messages"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            # Keep only last 50 messages
            if len(st.session_state["verbose_messages"]) > 50:
                st.session_state["verbose_messages"] = st.session_state["verbose_messages"][-50:]
            
            # Update the verbose display in real-time
            with verbose_placeholder.container():
                for msg in st.session_state["verbose_messages"][-20:]:  # Show last 20 messages
                    st.text(msg)
            
        
        # Initialize verbose messages
        verbose_callback(f"Starting analysis for {symbol}...")
        
        # Run analysis directly with progress updates
        try:
            result = run_crewai_analysis(symbol, openai_api_key, update_progress, verbose_callback)
            
            # Store in session state
            st.session_state["analysis_result"] = result
            st.session_state["analysis_running"] = False  # Clear running flag
            verbose_callback("Analysis completed and results stored!")
            
            # Display results immediately
            with results_placeholder.container():
                if result.get("success", False):
                    st.success(f"✅ Analysis completed for {result.get('symbol', 'Unknown')}")
                    
                    # Display the result
                    st.subheader("📊 Comprehensive Analysis Report")
                    
                    # Format the result nicely
                    analysis_text = result.get("result", "")
                    
                    # Try to parse and display structured content
                    if analysis_text:
                        # If it's a string, display it directly
                        if isinstance(analysis_text, str):
                            st.markdown("### Full Analysis Report")
                            st.markdown(analysis_text)
                        else:
                            # If it's an object, try to display it nicely
                            st.json(analysis_text)
                    else:
                        st.warning("No analysis text found in result")
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"crewai_analysis_{result.get('symbol', 'stock')}_{timestamp}.txt"
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=str(analysis_text),
                        file_name=filename,
                        mime="text/plain"
                    )
                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            verbose_callback(f"Analysis failed with error: {str(e)}")
            st.session_state["analysis_result"] = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol
            }
            st.session_state["analysis_running"] = False  # Clear running flag
            
            # Display error in results section
            with results_placeholder.container():
                st.error(f"❌ Analysis failed: {str(e)}")
        
        # Clear only the progress container after completion, keep verbose log
        with progress_container:
            st.empty()  # Clear progress elements
    
    # Show verbose messages if they exist (for completed analyses)
    if "analysis_result" in st.session_state and "verbose_messages" in st.session_state and st.session_state["verbose_messages"]:
        st.subheader("🤖 Agent Activity Log")
        for msg in st.session_state["verbose_messages"][-20:]:  # Show last 20 messages
            st.text(msg)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 CrewAI + MCP Stocks Analysis Demo | Powered by OpenAI & Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
