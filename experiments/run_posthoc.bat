@echo off
REM Runs TRI_FRAME_posthoc.py (per-feature attribution + hypervolume) and restarts it if it stops early.
REM Needs TRI_FRAME_full.py in the same folder. Start from the Anaconda Prompt.
cd /d "%~dp0"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
for /L %%i in (1,1,30) do (
  echo ==== attempt %%i started %date% %time% ==== >> run_posthoc_console.log
  python -u TRI_FRAME_posthoc.py --part both --hv-runs 25 >> run_posthoc_console.log 2>&1
  if not errorlevel 1 goto finished
  echo ==== attempt %%i stopped with exit code %errorlevel% at %date% %time%, restarting in 30 s ==== >> run_posthoc_console.log
  timeout /t 30 /nobreak >nul
)
echo Gave up after 30 attempts. Send run_posthoc_console.log for diagnosis.
goto end
:finished
echo ==== POSTHOC FINISHED %date% %time% ==== >> run_posthoc_console.log
echo Post-hoc analysis finished.
:end
pause
