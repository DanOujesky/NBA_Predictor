@echo off
cd /d %~dp0
title NBA Predictor

if not exist .venv (
    echo Setting up environment for first time...
    py -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python not found. Install Python 3.10+ from python.org
        pause
        exit /b 1
    )
    call .venv\Scripts\activate
    echo Installing packages ^(this takes a few minutes on first run^)...
    pip install --quiet -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Package installation failed.
        pause
        exit /b 1
    )
) else (
    call .venv\Scripts\activate
    py -c "import lightgbm, xgboost, flask, nba_api, cloudscraper" 2>nul || (
        echo Updating packages...
        pip install --quiet -r requirements.txt
    )
)

echo Starting NBA Predictor...
py main.py
