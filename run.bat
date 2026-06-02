@echo off
echo Starting Streamlit app using the project's virtual environment...
.\venv\Scripts\python.exe -m streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Failed to start the app. Please make sure the virtual environment exists and is populated.
    pause
)
