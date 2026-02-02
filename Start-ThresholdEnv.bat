@echo off
REM Launch a new CMD in the project folder with the venv activated

REM Move to the folder where this .bat lives (the project root)
pushd "%~dp0"

REM Start a persistent shell, activate venv, set a nice title
cmd /k "call .venv\Scripts\activate.bat & title threshold-completeness (venv)"