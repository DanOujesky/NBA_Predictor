@echo off
cd /d %~dp0

if not exist .venv (
    py -m venv .venv
)
call .venv\Scripts\activate

pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean NBAPredictor.spec

echo.
if not exist "dist\NBAPredictor\NBAPredictor.exe" (
    echo BUILD FAILED - viz vystup nahore
    pause
    exit /b 1
)

echo Kopiruji data do dist...
if exist "storage\raw" (
    xcopy /E /I /Y "storage\raw" "dist\NBAPredictor\storage\raw" >nul
    echo   - raw data OK
)
if exist "storage\processed" (
    xcopy /E /I /Y "storage\processed" "dist\NBAPredictor\storage\processed" >nul
    echo   - processed data OK
)
if exist "storage\trained" (
    xcopy /E /I /Y "storage\trained" "dist\NBAPredictor\storage\trained" >nul
    echo   - model OK
)

echo.
echo Build OK - dist\NBAPredictor\NBAPredictor.exe
pause
