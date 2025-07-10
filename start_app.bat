@echo off
echo ========================================
echo    Quant Trader Pro - Windows Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking dependencies...
echo.

REM Check if requirements are installed
python -c "import streamlit, plotly, yfinance, pandas, ta, numpy, alpaca_trade_api" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo.
)

echo Starting Quant Trader Pro...
echo.
echo The application will open in your default web browser.
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application.
echo ========================================
echo.

REM Start the application
python run_app.py

pause 