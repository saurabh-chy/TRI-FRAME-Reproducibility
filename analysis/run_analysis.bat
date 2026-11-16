@echo off
REM Rebuilds the statistics, the paper's tables and figures, and the supplementary material.
REM Usage (Anaconda Prompt, inside this folder):  run_analysis.bat C:\path\to\output\full
set R=%1
if "%R%"=="" set R=..\output\full
set OUT=%~dp0..\reproduced
mkdir "%OUT%" 2>nul
python "%~dp0statistical_tests.py" "%R%" "%OUT%\stats" || goto err
python "%~dp0make_figures_tables.py" "%R%" "%OUT%\exhibits" "%OUT%\stats" || goto err
if exist "%R%\bounds" python "%~dp0bounds_analysis.py" "%R%" "%OUT%\bounds" || goto err
python "%~dp0make_supplement.py" "%R%" "%OUT%\supplementary_material.md" "%OUT%\stats" || goto err
where pandoc >nul 2>nul && pandoc "%OUT%\supplementary_material.md" -o "%OUT%\supplementary_material.docx"
echo Done: %OUT%
goto end
:err
echo Analysis failed.
:end
