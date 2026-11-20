@echo off
setlocal
if exist C:\mingw64\bin\gcc.exe set "PATH=C:\mingw64\bin;%PATH%"
mingw32-make -C "%~dp0.." native
exit /b %ERRORLEVEL%
