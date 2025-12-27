@echo off
echo ========================================
echo   DeepSeek-OCR Web Application
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist "env\Scripts\activate.bat" (
    echo Activating virtual environment...
    call env\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found!
    echo Please create one first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
)

echo.
echo Starting server...
echo Open http://localhost:5000 in your browser
echo Press Ctrl+C to stop
echo.

python app.py

pause

