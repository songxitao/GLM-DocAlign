@echo off
echo ===================================================
echo   Starting official MinerU Pipeline models download...
echo   Only downloading lightweight models (CPU/Low VRAM)
echo   Skipping heavy VLM models to save disk space
echo ===================================================

"E:\conda\envs\mineru\Scripts\mineru-models-download.exe" -s modelscope -m pipeline

echo ===================================================
echo   All pipeline models successfully aligned and configured!
echo ===================================================
pause
