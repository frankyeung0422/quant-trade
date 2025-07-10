#!/usr/bin/env python3
"""
Quant Trader Pro - Application Launcher
A simple launcher script for the Quant Trader Pro web interface.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'yfinance',
        'pandas',
        'ta',
        'numpy',
        'alpaca_trade_api'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_alpaca_credentials():
    """Check if Alpaca API credentials are configured."""
    api_key = os.getenv('APCA_API_KEY_ID')
    api_secret = os.getenv('APCA_API_SECRET_KEY')
    
    if not api_key or not api_secret:
        print("⚠️  Alpaca API credentials not found.")
        print("   Live trading features will be disabled.")
        print("\n🔑 To enable live trading, set environment variables:")
        print("   export APCA_API_KEY_ID='your_api_key'")
        print("   export APCA_API_SECRET_KEY='your_secret_key'")
        print("   export APCA_API_BASE_URL='https://paper-api.alpaca.markets'")
        return False
    
    print("✅ Alpaca API credentials found!")
    return True

def main():
    """Main launcher function."""
    print("🚀 Quant Trader Pro - Application Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check Alpaca credentials
    print("\n🔑 Checking Alpaca credentials...")
    check_alpaca_credentials()
    
    # Check if web_app.py exists
    if not os.path.exists('web_app.py'):
        print("❌ web_app.py not found!")
        print("   Make sure you're running this script from the project directory.")
        sys.exit(1)
    
    print("\n🌐 Starting Quant Trader Pro web interface...")
    print("   The application will open in your default web browser.")
    print("   URL: http://localhost:8501")
    print("\n   Press Ctrl+C to stop the application.")
    print("=" * 50)
    
    try:
        # Start Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'web_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 