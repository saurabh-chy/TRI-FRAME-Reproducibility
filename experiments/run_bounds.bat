@echo off
REM Per-group flip rates, exact worst-case fairness gap and certified bounds (Proposition 1).
REM Needs TRI_FRAME_full.py and TRI_FRAME_bounds.py in the same folder. Start from the Anaconda Prompt.
REM Resumable: finished units are skipped, so it continues where it stopped.
cd /d "%~dp0"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
for /L %%i in (1,1,30) do (
  echo ==== attempt %%i started %date% %time% ==== >> run_bounds_console.log
  python -u TRI_FRAME_bounds.py --part both >> run_bounds_console.log 2>&1
  if not errorlevel 1 goto finished
  echo ==== attempt %%i stopped with exit code %errorlevel% at %date% %time%, restarting in 30 s ==== >> run_bounds_console.log
  timeout /t 30 /nobreak >nul
)
echo Gave up after 30 attempts. Send run_bounds_console.log for diagnosis.
goto end
:finished
echo ==== BOUNDS FINISHED %date% %time% ==== >> run_bounds_console.log
echo Bounds analysis finished.
:end
pause
