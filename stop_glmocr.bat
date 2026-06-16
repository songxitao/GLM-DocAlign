@echo off
chcp 65001 > nul

:: ============================================================
::      GLM-OCR Service Shutdown
:: ============================================================
echo.

echo [1/2] Stopping Gradio frontend...
taskkill /FI "WINDOWTITLE eq GLM_OCR_FRONTEND_PROCESS" /F /T >nul 2>&1
if %errorlevel% equ 0 (
    echo    [OK] Frontend stopped.
) else (
    echo    [INFO] No running frontend found.
)
echo.

echo [2/2] Stopping Docker container 'glm-ocr-service'...
docker rm -f glm-ocr-service >nul 2>&1
if %errorlevel% neq 0 (
    echo    [WARN] Failed. Container may already be stopped.
) else (
    echo    [OK] Backend stopped.
)
echo.

echo ============================================================
echo           All GLM-OCR services stopped.
echo ============================================================
echo.
pause
