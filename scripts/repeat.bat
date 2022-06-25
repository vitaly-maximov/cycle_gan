@echo off

:command
cmd /c %1

if %errorlevel%==0 goto end

timeout %2
goto command

:end