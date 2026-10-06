@echo off
cd /d "%~dp0"
python -m floodsim %*
if errorlevel 1 pause
