@echo off
setlocal
cd /d "%~dp0"

set "APP_NAME=FNN_Fault_Diagnosis_GUI"
set "GUI_SCRIPT=fault_diagnosis_gui.py"
set "PT_FILE=fnn_fault_checkpoint.pt"
set "OUTPUT_DIR=exe_folder"
set "SMOKE_LOG=FNN_Fault_Diagnosis_GUI_smoke_test_error.txt"
set "VENV_DIR=.venv_gui_build"

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

echo [1/4] Preparing a clean build environment
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %SYSTEM_PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
)
set "BUILD_PYTHON=%VENV_DIR%\Scripts\python.exe"
"%BUILD_PYTHON%" --version
if errorlevel 1 goto :fail

echo [2/4] Installing build packages
"%BUILD_PYTHON%" -m ensurepip --upgrade >nul 2>nul
"%BUILD_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%BUILD_PYTHON%" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
"%BUILD_PYTHON%" -m pip install --upgrade numpy matplotlib PyQt5 pyinstaller pyinstaller-hooks-contrib
if errorlevel 1 goto :fail

echo [3/4] Building onefile EXE
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
  --collect-all torch ^
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

echo [4/4] Running the bundled .pt inference self-test
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

