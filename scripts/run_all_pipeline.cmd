@echo off
cd /d "%~dp0.."

python scripts\02_ablation_tests.py --experiment-root results\experiments\complete_experiment
if errorlevel 1 goto :error

python scripts\02b_aggregate_ablation_results.py
if errorlevel 1 goto :error

python scripts\03_evaluate_nvchvc_station_scramble.py
if errorlevel 1 goto :error

python scripts\04_zero_shot_cross_volcano.py
if errorlevel 1 goto :error

python scripts\05_progressive_finetuning.py
if errorlevel 1 goto :error

echo.
echo Pipeline completed successfully.
exit /b 0

:error

echo.
echo Pipeline failed at a step above.
exit /b 1
