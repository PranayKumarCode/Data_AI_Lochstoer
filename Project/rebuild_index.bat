@echo off
REM Rebuild PDF Index - One-Click Indexer
echo ========================================
echo   Rebuilding Lecture Index
echo ========================================
echo.

REM Change to project directory
cd /d "C:\Users\Pranay Kumar\Documents\Python\DataAI"

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Build index
python build_index_v2.py

echo.
echo ========================================
echo   Index Rebuild Complete!
echo ========================================
pause