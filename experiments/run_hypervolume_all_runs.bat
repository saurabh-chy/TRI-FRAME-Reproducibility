@echo off
REM Extends the hypervolume analysis to all runs: 10 hold-out runs (Adult, ACS Income, COMPAS)
REM and all 25 folds (South German Credit). Finished units are skipped, so it resumes where it stopped.
REM Needs TRI_FRAME_full.py and TRI_FRAME_posthoc.py in the same folder. Start from the Anaconda Prompt.
cd /d "%~dp0"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
for /L %%i in (1,1,30) do (
  echo ==== attempt %%i started %date% %time% ==== >> run_hypervolume_console.log
  python -u TRI_FRAME_posthoc.py --part hv --hv-runs 25 >> run_hypervolume_console.log 2>&1
  if not errorlevel 1 goto finished
  echo ==== attempt %%i stopped with exit code %errorlevel% at %date% %time%, restarting in 30 s ==== >> run_hypervolume_console.log
  timeout /t 30 /nobreak >nul
)
echo Gave up after 30 attempts. Send run_hypervolume_console.log for diagnosis.
goto end
:finished
echo ==== HYPERVOLUME (ALL RUNS) FINISHED %date% %time% ==== >> run_hypervolume_console.log
echo Hypervolume analysis on all runs finished.
:end
pause
