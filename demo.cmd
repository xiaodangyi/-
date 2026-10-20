@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
"C:\Users\LENOVO\Documents\Codex\2026-10-18\gen\anaconda-envs\ai-cuda\python.exe" -s "scripts\demo.py" %*
