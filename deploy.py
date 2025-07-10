#!/usr/bin/env python3
"""
Deployment script for Quant Trader Pro
Validates configuration and provides deployment instructions
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'yfinance', 
        'ta', 'numpy', 'alpaca_trade_api', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def check_environment_variables():
    """Check if required environment variables are set."""
    print("\n🔍 Checking environment variables...")
    
    required_vars = [
        'APCA_API_KEY_ID',
        'APCA_API_SECRET_KEY'
    ]
    
    optional_vars = [
        'APCA_API_BASE_URL',
        'STREAMLIT_PORT',
        'STREAMLIT_HOST'
    ]
    
    missing_vars = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            missing_vars.append(var)
            print(f"❌ {var}")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} (optional)")
        else:
            print(f"ℹ️ {var} (optional, using default)")
    
    if missing_vars:
        print(f"\n⚠️ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables before deployment.")
        return False
    
    print("✅ Environment variables are configured!")
    return True

def check_files():
    """Check if all required files exist."""
    print("\n🔍 Checking required files...")
    
    required_files = [
        'web_app.py',
        'requirements.txt',
        '.streamlit/config.toml',
        'packages.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files exist!")
    return True

def test_streamlit_app():
    """Test if the Streamlit app can be imported."""
    print("\n🔍 Testing Streamlit app...")
    
    try:
        # Try to import the main app
        import web_app
        print("✅ Streamlit app imports successfully!")
        return True
    except Exception as e:
        print(f"❌ Error importing Streamlit app: {str(e)}")
        return False

def show_deployment_instructions():
    """Show deployment instructions."""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n1. STREAMLIT CLOUD (Recommended):")
    print("   - Push your code to GitHub")
    print("   - Go to https://share.streamlit.io")
    print("   - Connect your GitHub repository")
    print("   - Set main file to: web_app.py")
    print("   - Add environment variables in the dashboard")
    print("   - Deploy!")
    
    print("\n2. HEROKU:")
    print("   - Install Heroku CLI")
    print("   - Run: heroku create your-app-name")
    print("   - Set environment variables:")
    print("     heroku config:set APCA_API_KEY_ID=your_key")
    print("     heroku config:set APCA_API_SECRET_KEY=your_secret")
    print("   - Run: git push heroku main")
    
    print("\n3. LOCAL DEPLOYMENT:")
    print("   - Run: streamlit run web_app.py")
    print("   - App will be available at: http://localhost:8501")
    
    print("\n4. ENVIRONMENT VARIABLES:")
    print("   Required:")
    print("   - APCA_API_KEY_ID: Your Alpaca API key")
    print("   - APCA_API_SECRET_KEY: Your Alpaca secret key")
    print("   Optional:")
    print("   - APCA_API_BASE_URL: https://paper-api.alpaca.markets (default)")
    print("   - STREAMLIT_PORT: 8501 (default)")
    print("   - STREAMLIT_HOST: localhost (default)")

def main():
    """Main deployment validation function."""
    print("🔧 Quant Trader Pro - Deployment Validation")
    print("="*50)
    
    # Run all checks
    checks = [
        check_dependencies(),
        check_environment_variables(),
        check_files(),
        test_streamlit_app()
    ]
    
    if all(checks):
        print("\n🎉 All checks passed! Your app is ready for deployment.")
        show_deployment_instructions()
    else:
        print("\n❌ Some checks failed. Please fix the issues above before deploying.")
        sys.exit(1)

if __name__ == "__main__":
    main() 