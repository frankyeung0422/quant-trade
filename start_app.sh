#!/bin/bash

echo "========================================"
echo "   Quant Trader Pro - Unix Launcher"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 from https://python.org"
    exit 1
fi

echo "Python found. Checking dependencies..."
echo

# Check if requirements are installed
python3 -c "import streamlit, plotly, yfinance, pandas, ta, numpy, alpaca_trade_api" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install requirements"
        exit 1
    fi
    echo
fi

echo "Starting Quant Trader Pro..."
echo
echo "The application will open in your default web browser."
echo "URL: http://localhost:8501"
echo
echo "Press Ctrl+C to stop the application."
echo "========================================"
echo

# Start the application
python3 run_app.py 