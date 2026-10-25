@echo off
setlocal
cd /d "%~dp0"

set "APP_NAME=FNN_Fault_Diagnosis_GUI"
set "GUI_SCRIPT=fault_diagnosis_gui.py"
set "PT_FILE=fnn_fault_checkpoint.pt"
set "OUTPUT_DIR=exe_folder"
set "SMOKE_LOG=FNN_Fault_Diagnosis_GUI_smoke_test_error.txt"
set "BOOTSTRAP_VENV=.venv_gui_bootstrap"
set "MANAGED_PYTHON_DIR=.python_gui_build"
set "VENV_DIR=.venv_gui_build_py312"

echo ========================================
echo FNN Fault Diagnosis - onefile EXE build
echo ========================================
echo.

for %%F in ("%GUI_SCRIPT%" "%PT_FILE%") do (
    if not exist "%%~F" (
        echo ERROR: Missing required file: %%~F
        goto :fail
    )
)

echo [1/5] Locating the system Python
set "SYSTEM_PYTHON="
where py >nul 2>nul
if not errorlevel 1 set "SYSTEM_PYTHON=py -3"
if not defined SYSTEM_PYTHON (
    where python >nul 2>nul
    if not errorlevel 1 set "SYSTEM_PYTHON=python"
)
if not defined SYSTEM_PYTHON (
    echo ERROR: Python 3 is not installed.
    goto :fail
)

echo [2/5] Preparing the managed Python 3.12 runtime
set "PYTHON312="
for /f "delims=" %%F in ('dir /b /s "%MANAGED_PYTHON_DIR%\python.exe" 2^>nul') do if not defined PYTHON312 set "PYTHON312=%%F"

if not defined PYTHON312 (
    if not exist "%BOOTSTRAP_VENV%\Scripts\python.exe" (
        %SYSTEM_PYTHON% -m venv "%BOOTSTRAP_VENV%"
        if errorlevel 1 goto :fail
    )
    "%BOOTSTRAP_VENV%\Scripts\python.exe" -m pip install --upgrade pip uv
    if errorlevel 1 goto :fail
    "%BOOTSTRAP_VENV%\Scripts\uv.exe" python install 3.12 --install-dir "%MANAGED_PYTHON_DIR%"
    if errorlevel 1 goto :fail
    for /f "delims=" %%F in ('dir /b /s "%MANAGED_PYTHON_DIR%\python.exe" 2^>nul') do if not defined PYTHON312 set "PYTHON312=%%F"
)

if not defined PYTHON312 (
    echo ERROR: Managed Python 3.12 was not installed.
    goto :fail
)
"%PYTHON312%" --version
if errorlevel 1 goto :fail

echo [3/5] Preparing a clean build environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
    "%PYTHON312%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
)
set "BUILD_PYTHON=%VENV_DIR%\Scripts\python.exe"
"%BUILD_PYTHON%" --version
if errorlevel 1 goto :fail

echo [4/5] Installing packages and building the onefile EXE
"%BUILD_PYTHON%" -m ensurepip --upgrade >nul 2>nul
"%BUILD_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%BUILD_PYTHON%" -m pip install --upgrade "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
"%BUILD_PYTHON%" -m pip install --upgrade "numpy==1.26.4" matplotlib PyQt5 pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 goto :fail

if exist "build" rmdir /s /q "build"
if exist "%OUTPUT_DIR%\%APP_NAME%.exe" del /q "%OUTPUT_DIR%\%APP_NAME%.exe"
if exist "%APP_NAME%.spec" del /q "%APP_NAME%.spec"

"%BUILD_PYTHON%" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name "%APP_NAME%" ^
  --distpath "%OUTPUT_DIR%" ^
  --add-data "%PT_FILE%;." ^
  --collect-binaries torch ^
  --hidden-import torch ^
  --hidden-import torch.nn ^
  --hidden-import matplotlib.backends.backend_qt5agg ^
  --collect-submodules numpy._core ^
  --exclude-module PyQt6 ^
  --exclude-module PySide6 ^
  --exclude-module PySide2 ^
  "%GUI_SCRIPT%"
if errorlevel 1 goto :fail

if not exist "%OUTPUT_DIR%\%APP_NAME%.exe" (
    echo ERROR: EXE was not created.
    goto :fail
)

echo [5/5] Running the bundled .pt inference self-test
if exist "%OUTPUT_DIR%\%SMOKE_LOG%" del /q "%OUTPUT_DIR%\%SMOKE_LOG%"
powershell.exe -NoProfile -Command "$p = Start-Process -FilePath '.\%OUTPUT_DIR%\%APP_NAME%.exe' -ArgumentList '--smoke-test' -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode"
if errorlevel 1 (
    echo ERROR: The generated EXE failed its self-test.
    if exist "%OUTPUT_DIR%\%SMOKE_LOG%" type "%OUTPUT_DIR%\%SMOKE_LOG%"
    goto :fail
)

echo.
echo Build completed:
echo %CD%\%OUTPUT_DIR%\%APP_NAME%.exe
echo.
echo The .pt file and PyTorch are inside the onefile EXE.
echo Only the EXE is required on another 64-bit Windows PC.
echo.
pause
exit /b 0

:fail
echo.
echo Build failed. Review the ERROR message above.
echo.
pause
exit /b 1

