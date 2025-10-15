#!/usr/bin/env python3
"""
Streamlit CrewAI Demo Runner

This script provides an easy way to run the Streamlit CrewAI demo.
It automatically handles MCP server startup and provides helpful instructions.
"""

import os
import sys
import subprocess
import time
import requests
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'crewai',
        'fastapi',
        'uvicorn',
        'requests',
        'pandas',
        'yfinance'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_mcp_server():
    """Check if MCP server is running"""
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_mcp_server():
    """Start MCP server in background"""
    print("🚀 Starting MCP server...")
    try:
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "127.0.0.1", 
            "--port", "8001",
            "--log-level", "warning"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        for i in range(20):
            time.sleep(0.5)
            if check_mcp_server():
                print("✅ MCP server started successfully")
                return process
            print(f"   Waiting for server... ({i+1}/20)")
        
        print("❌ Failed to start MCP server")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Error starting MCP server: {e}")
        return None

def main():
    """Main runner function"""
    print("🎯 Streamlit CrewAI + MCP Stocks Analysis Demo")
    print("=" * 60)
    
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All requirements satisfied")
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("\n🔑 OpenAI API Key not found in environment variables.")
        print("You can set it with:")
        print("   export OPENAI_API_KEY='your-key-here'")
        print("Or enter it directly in the Streamlit app.")
    else:
        print(f"✅ OpenAI API key found: {openai_key[:8]}...")
    
    # Check MCP server
    print("\n🔗 Checking MCP server...")
    server_process = None
    
    if not check_mcp_server():
        print("MCP server not running, starting it...")
        server_process = start_mcp_server()
        if not server_process:
            print("❌ Could not start MCP server. Please start it manually:")
            print("   uvicorn api:app --host 127.0.0.1 --port 8001")
            sys.exit(1)
    else:
        print("✅ MCP server is already running")
    
    # Start Streamlit
    print("\n🌐 Starting Streamlit app...")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the demo.")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_crewai_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping demo...")
        
    finally:
        # Clean up MCP server if we started it
        if server_process:
            print("🧹 Stopping MCP server...")
            server_process.terminate()
            server_process.wait()
            print("✅ Demo stopped")

if __name__ == "__main__":
    main()
