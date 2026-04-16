@echo off
cd /d %~dp0

if not exist .venv (
    py -m venv .venv
)
call .venv\Scripts\activate

pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean vendor\build\NBAPredictor.spec
if errorlevel 1 (
    echo.
    echo BUILD FAILED - viz chyby nahore
    pause
    exit /b 1
)

if not exist "dist\NBAPredictor\NBAPredictor.exe" (
    echo BUILD FAILED - NBAPredictor.exe nebyl vytvoren
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

echo Balim do ZIP...
where powershell >nul 2>&1
if %errorlevel% == 0 (
    powershell -Command "Compress-Archive -Path 'dist\NBAPredictor' -DestinationPath 'dist\NBAPredictor.zip' -Force"
    if exist "dist\NBAPredictor.zip" (
        for %%A in ("dist\NBAPredictor.zip") do echo ZIP hotov: %%~zA bytes - dist\NBAPredictor.zip
    )
) else (
    echo PowerShell neni dostupny - zkopiruj slozku dist\NBAPredictor rucne
)

echo.
echo === CO UDELAT DU SKOLY ===
echo Moznost A ^(USB^): Zkopiruj slozku dist\NBAPredictor na USB flash disk
echo Moznost B ^(siti^): Nahraj dist\NBAPredictor.zip na Google Drive / GitHub
echo Ve skole: rozbal ZIP -^> spust NBAPredictor.exe
echo.
pause
