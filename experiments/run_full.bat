@echo off
REM Runs TRI_FRAME_full.py and restarts it automatically if it stops before finishing.
REM Finished models are skipped on every restart. Start this from the Anaconda Prompt.
cd /d "%~dp0"
REM Keep the computer awake while plugged in (restore later in Settings > Power if you want).
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
for /L %%i in (1,1,30) do (
  echo ==== attempt %%i started %date% %time% ==== >> run_full_console.log
  python -u TRI_FRAME_full.py >> run_full_console.log 2>&1
  if not errorlevel 1 goto finished
  echo ==== attempt %%i stopped with exit code %errorlevel% at %date% %time%, restarting in 30 s ==== >> run_full_console.log
  timeout /t 30 /nobreak >nul
)
echo Gave up after 30 attempts. Send run_full_console.log for diagnosis.
goto end
:finished
echo ==== FULL RUN FINISHED %date% %time% ==== >> run_full_console.log
echo Full run finished.
:end
pause
