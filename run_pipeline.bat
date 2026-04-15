@echo off
cd /d %~dp0

if not exist .venv (
    py -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt

echo.
echo  1) Full pipeline  (download all data + train + predict)
echo  2) Quick update   (current season only, no bref re-scrape)
echo  3) Train only     (retrain on existing features.csv)
echo  4) Serve app      (start web server, no pipeline)
echo.
set /p CHOICE="Select option [1-4]: "

if "%CHOICE%"=="1" (
    py pipeline.py
) else if "%CHOICE%"=="2" (
    py pipeline.py --quick-update
) else if "%CHOICE%"=="3" (
    py pipeline.py --train-only
) else if "%CHOICE%"=="4" (
    py pipeline.py --train-only --serve
) else (
    echo Invalid choice, running quick update by default
    py pipeline.py --quick-update
)

pause
