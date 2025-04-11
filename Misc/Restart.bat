@echo off
setlocal

cd /d %~dp0%
echo %CD%

:: Reads restart count from cache.txt
if EXIST cache.txt (
    for /f "tokens=2 delims==" %%A in ('findstr /R "^RESTARTS=" cache.txt') do set "RESTARTS=%%A"
) else (
    set RESTARTS=0
    echo RESTARTS=0 > cache.txt
)
set /a RESTARTS=%RESTARTS%+1

:: After 5 unsuccessful restarts restart pc !
if %RESTARTS% GEQ 5 (
    ECHO RESTARTS=0 > cache.txt
    shutdown /r /t 1
    exit
)

:: Save restart amount to the cache file
ECHO RESTARTS=%RESTARTS% > cache.txt
 
:: Kill all other mains
taskkill /FI "WINDOWTITLE eq PeopleDetection" /F

:: Go to the root folder
cd ..
:: Start main again
Scripts\python.exe main.py --deploy 1 --restart 1 2>> errors

endlocal