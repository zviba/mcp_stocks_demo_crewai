#!/usr/bin/env python3
"""
CrewAI + MCP Stocks Demo

This script demonstrates how to use CrewAI with the MCP stocks server
to perform comprehensive stock analysis using specialized agents.

The workflow:
1. Research Agent: Searches for stock symbols and gathers basic data
2. Technical Analyst: Performs technical analysis using indicators and events
3. Report Writer: Synthesizes findings into a comprehensive report

Usage:
    python crewai_stocks_demo.py --symbol AAPL --openai-api-key YOUR_KEY
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List
import requests
from datetime import datetime

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# MCP Server Configuration
MCP_SERVER_URL = "http://127.0.0.1:8001"

class MCPTool(BaseTool):
    """Base class for MCP server tools"""
    
    def __init__(self, endpoint: str, name: str, description: str):
        self.endpoint = endpoint
        super().__init__(name=name, description=description)
    
    def _run(self, **kwargs) -> str:
        """Execute the MCP tool via HTTP request"""
        try:
            response = requests.post(
                f"{MCP_SERVER_URL}{self.endpoint}",
                json=kwargs,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error calling {self.name}: {str(e)}"

class SearchSymbolsTool(MCPTool):
    """Tool for searching stock symbols"""
    
    def __init__(self):
        super().__init__(
            endpoint="/search",
            name="search_symbols",
            description="Search for stock symbols by company name or ticker"
        )

class GetQuoteTool(MCPTool):
    """Tool for getting latest stock quote"""
    
    def __init__(self):
        super().__init__(
            endpoint="/quote",
            name="get_quote",
            description="Get latest price, change percentage, and volume for a stock"
        )

class GetPriceSeriesTool(MCPTool):
    """Tool for getting historical price data"""
    
    def __init__(self):
        super().__init__(
            endpoint="/series",
            name="get_price_series",
            description="Get historical OHLCV price data for a stock"
        )

class GetIndicatorsTool(MCPTool):
    """Tool for getting technical indicators"""
    
    def __init__(self):
        super().__init__(
            endpoint="/indicators",
            name="get_indicators",
            description="Get technical indicators (SMA, EMA, RSI) for a stock"
        )

class GetEventsTool(MCPTool):
    """Tool for detecting market events"""
    
    def __init__(self):
        super().__init__(
            endpoint="/events",
            name="get_events",
            description="Detect market events like gaps, volatility spikes, and 52-week extremes"
        )

class GetExplanationTool(MCPTool):
    """Tool for getting LLM explanation of technical analysis"""
    
    def __init__(self):
        super().__init__(
            endpoint="/explain",
            name="get_explanation",
            description="Get AI-powered explanation of technical analysis with market context"
        )

def check_mcp_server() -> bool:
    """Check if MCP server is running"""
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def create_agents(openai_api_key: str) -> Dict[str, Agent]:
    """Create specialized CrewAI agents"""
    
    # Initialize MCP tools
    search_tool = SearchSymbolsTool()
    quote_tool = GetQuoteTool()
    series_tool = GetPriceSeriesTool()
    indicators_tool = GetIndicatorsTool()
    events_tool = GetEventsTool()
    explanation_tool = GetExplanationTool()
    
    # Research Agent - Gathers basic information and data
    research_agent = Agent(
        role="Stock Research Specialist",
        goal="Gather comprehensive basic information about stocks including current quotes, historical data, and company details",
        backstory="""You are an experienced stock researcher with deep knowledge of financial markets. 
        Your expertise lies in efficiently gathering and organizing stock data from multiple sources. 
        You excel at finding relevant information quickly and presenting it in a clear, structured format.""",
        tools=[search_tool, quote_tool, series_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # Technical Analyst - Performs technical analysis
    technical_agent = Agent(
        role="Technical Analysis Expert",
        goal="Perform detailed technical analysis using indicators, patterns, and market events to assess stock momentum and trends",
        backstory="""You are a seasoned technical analyst with 15+ years of experience in chart analysis and market indicators. 
        You specialize in interpreting technical signals, identifying patterns, and understanding market psychology. 
        Your analysis is methodical and based on proven technical analysis principles.""",
        tools=[indicators_tool, events_tool, explanation_tool],
        verbose=True,
        allow_delegation=False
    )
    
    # Report Writer - Synthesizes findings
    report_agent = Agent(
        role="Financial Report Writer",
        goal="Create comprehensive, well-structured investment reports that synthesize research and technical analysis into actionable insights",
        backstory="""You are a professional financial writer with expertise in translating complex market data into clear, 
        actionable reports. You have a talent for presenting technical information in an accessible way while maintaining 
        accuracy and professional standards. Your reports are known for their clarity and practical insights.""",
        tools=[],  # No tools needed - focuses on synthesis
        verbose=True,
        allow_delegation=False
    )
    
    return {
        "research": research_agent,
        "technical": technical_agent,
        "report": report_agent
    }

def create_tasks(symbol: str, openai_api_key: str) -> List[Task]:
    """Create tasks for the crew workflow"""
    
    # Task 1: Research and Data Gathering
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
        agent=None,  # Will be assigned later
        tools=[SearchSymbolsTool(), GetQuoteTool(), GetPriceSeriesTool()]
    )
    
    # Task 2: Technical Analysis
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
        agent=None,  # Will be assigned later
        tools=[GetIndicatorsTool(), GetEventsTool(), GetExplanationTool()],
        context=[research_task]  # Depends on research task
    )
    
    # Task 3: Final Report Synthesis
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
        agent=None,  # Will be assigned later
        context=[research_task, technical_task]  # Depends on both previous tasks
    )
    
    return [research_task, technical_task, report_task]

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="CrewAI + MCP Stocks Analysis Demo")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to analyze (default: AAPL)")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key for LLM explanations")
    parser.add_argument("--mcp-url", default=MCP_SERVER_URL, help="MCP server URL (default: http://127.0.0.1:8001)")
    
    args = parser.parse_args()
    
    # Update global MCP server URL if provided
    global MCP_SERVER_URL
    MCP_SERVER_URL = args.mcp_url
    
    print("🚀 CrewAI + MCP Stocks Analysis Demo")
    print("=" * 50)
    
    # Check MCP server availability
    print(f"📡 Checking MCP server at {MCP_SERVER_URL}...")
    if not check_mcp_server():
        print("❌ MCP server is not running!")
        print("Please start the MCP server first:")
        print("  uvicorn api:app --host 127.0.0.1 --port 8001")
        sys.exit(1)
    print("✅ MCP server is running")
    
    # Create agents
    print(f"\n👥 Creating specialized agents for {args.symbol} analysis...")
    agents = create_agents(args.openai_api_key)
    
    # Create tasks
    print("📋 Setting up analysis tasks...")
    tasks = create_tasks(args.symbol, args.openai_api_key)
    
    # Assign agents to tasks
    tasks[0].agent = agents["research"]
    tasks[1].agent = agents["technical"]
    tasks[2].agent = agents["report"]
    
    # Create crew
    print("🏗️  Assembling analysis crew...")
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )
    
    # Execute analysis
    print(f"\n🔍 Starting comprehensive analysis of {args.symbol}...")
    print("This may take a few minutes as agents gather data and perform analysis...")
    
    try:
        result = crew.kickoff()
        
        print("\n" + "=" * 50)
        print("📊 ANALYSIS COMPLETE")
        print("=" * 50)
        print(result)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_analysis_{args.symbol}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Stock Analysis Report for {args.symbol}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(result))
        
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
