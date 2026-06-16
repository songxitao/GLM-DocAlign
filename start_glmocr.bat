@echo off
chcp 65001 > nul
setlocal

:: ===================================================================
::           GLM-OCR vLLM Service Launcher (v2.1)
:: ===================================================================
echo.

:: ================= CONFIG =================
set CACHE_DIR=D:\my_huggingface_cache
set LOCAL_MODEL_DIR=E:\project\GLM-OCR\model\GLM-OCR
set CONTAINER_NAME=glm-ocr-service
set IMAGE_NAME=my-vllm:glm-ocr
set PORT=8700
set CONDA_PATH=D:\program files\Miniconda
set CONDA_ENV=deepseek-ocr
:: ==========================================

:: --- Step 1: Conda ---
echo [1/4] Activating Conda env '%CONDA_ENV%'...
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"
if %errorlevel% neq 0 (
    echo    [FAIL] Cannot find Conda. Check CONDA_PATH.
    goto :error_exit
)
call conda activate %CONDA_ENV%
if %errorlevel% neq 0 (
    echo    [FAIL] Cannot activate Conda env '%CONDA_ENV%'.
    goto :error_exit
)
echo    [OK] Conda env activated.
echo.

:: --- Step 2: Docker Backend ---
echo [2/4] Starting Docker backend...
docker rm -f %CONTAINER_NAME% >nul 2>&1

start "GLM-OCR Backend Log" cmd /k docker run --gpus all -p %PORT%:%PORT% --name %CONTAINER_NAME% --ipc=host --shm-size=8g -v %CACHE_DIR%:/root/.cache/huggingface -v %LOCAL_MODEL_DIR%:/app/glm-ocr-model %IMAGE_NAME% --model /app/glm-ocr-model --served-model-name glm-ocr --trust-remote-code --allowed-local-media-path / --port %PORT% --max-model-len 16384 --gpu-memory-utilization 0.6 --limit-mm-per-prompt "{\"image\": 2}"

echo    [OK] Backend command sent. Check the new window.
echo.

:: --- Step 3: Frontend ---
echo [3/4] Starting Gradio frontend...
timeout /t 5 > nul
start "" http://127.0.0.1:7860

start "GLM_OCR_FRONTEND_PROCESS" cmd /c "python appocr_vllm_ui.py"
echo.

:: --- Step 4: Done ---
echo [4/4] Done!
echo ===================================================================
echo  Backend API:  http://127.0.0.1:%PORT%/v1/chat/completions
echo  Frontend UI:  http://127.0.0.1:7860
echo  Model name:   glm-ocr
echo.
echo  Wait for backend window to show "Uvicorn running on..." 
echo  before using the frontend.
echo.
echo  Use stop_glmocr.bat to shut down all services.
echo ===================================================================
echo.
goto :done

:error_exit
echo.
echo [ERROR] Startup failed. Press any key to exit...
pause > nul

:done
endlocal
exit /b
