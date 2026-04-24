@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Always run from repository root (folder containing this script)
cd /d "%~dp0"

set "ENV_NAME=paddle_env"
set "ENV_DIR=%CD%\%ENV_NAME%"
set "ACTIVATE_BAT=%ENV_DIR%\Scripts\activate.bat"

if exist "%ACTIVATE_BAT%" (
    echo Found virtual environment: %ENV_NAME%
) else (
    echo Virtual environment "%ENV_NAME%" was not found.
    choice /M "Do you want to create it now"
    if errorlevel 2 (
        echo Environment creation declined. Exiting.
        exit /b 1
    )

    echo Creating virtual environment "%ENV_NAME%"...
    py -3 -m venv "%ENV_NAME%" >nul 2>&1
    if errorlevel 1 (
        python -m venv "%ENV_NAME%"
        if errorlevel 1 (
            echo Failed to create virtual environment.
            exit /b 1
        )
    )

    call "%ACTIVATE_BAT%"
    if errorlevel 1 (
        echo Failed to activate virtual environment.
        exit /b 1
    )

    echo Installing dependencies...
    python -m pip install --upgrade pip
    if errorlevel 1 exit /b 1

    pip install --upgrade paddlepaddle
    if errorlevel 1 exit /b 1

    pip install "numpy<2"
    if errorlevel 1 exit /b 1

    pip install -r requirements.txt
    if errorlevel 1 exit /b 1
)

if not exist "%ACTIVATE_BAT%" (
    echo Activation script not found: %ACTIVATE_BAT%
    exit /b 1
)

call "%ACTIVATE_BAT%"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Starting training...
python tools/train.py ^
    --config configs/pp_liteseg/pp_liteseg_stdc2_vdd_512x512_200k.yml ^
    --resume_model output/iter_121000 ^
    --do_eval ^
    --use_vdl ^
    --num_workers 4 ^
    --batch_size 8 ^
    --save_interval 500 ^
    --save_dir output

set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" (
    echo Training exited with code %TRAIN_EXIT%.
    exit /b %TRAIN_EXIT%
)

echo Training finished successfully.
exit /b 0
