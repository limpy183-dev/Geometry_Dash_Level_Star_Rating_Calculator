@echo off
cd /d "%~dp0"
where python >nul 2>nul && (python app.py & goto done)
where py >nul 2>nul && (py -3 app.py & goto done)
echo Python 3.10 or newer is required: https://www.python.org/downloads/
:done
pause
