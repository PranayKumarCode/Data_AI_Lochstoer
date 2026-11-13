@echo off
REM Teach AI Examples - One-Click Trainer
echo ========================================
echo   Teaching AI System
echo ========================================
echo.

REM Change to project directory
cd /d "C:\Users\Pranay Kumar\Documents\Python\DataAI"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Run teaching script
python teach_ai.py

echo.
echo ========================================
echo   Training Complete!
echo ========================================
pause