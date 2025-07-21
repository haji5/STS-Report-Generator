@echo off
echo Report Generator - Python to EXE Compiler
echo ==========================================
echo.

REM Get the current directory (where the script is located)
set "SOURCE_DIR=%~dp0"

REM Ask user for output directory
set /p "OUTPUT_DIR=Enter the full path where you want to create the EXE (or press Enter for current directory): "

REM If no output directory specified, use current directory
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=%SOURCE_DIR%"

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo.
echo Compiling Python files to executable...
echo Source: %SOURCE_DIR%
echo Output: %OUTPUT_DIR%
echo.

REM Change to source directory
cd /d "%SOURCE_DIR%"

REM Run PyInstaller with the spec file
pyinstaller --distpath "%OUTPUT_DIR%" --workpath "%OUTPUT_DIR%\build" report_generator.spec

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✓ Compilation successful!
    echo ✓ Executable created: %OUTPUT_DIR%\ReportGenerator.exe
    echo ✓ Assets folder is embedded in the EXE
    echo.
    echo Cleaning up temporary files...
    if exist "%OUTPUT_DIR%\build" (
        rmdir /s /q "%OUTPUT_DIR%\build"
        echo ✓ Temporary build folder removed
    )
    echo.
    echo Your standalone executable is ready: %OUTPUT_DIR%\ReportGenerator.exe
) else (
    echo ✗ Compilation failed! Check the error messages above.
    echo Note: Build folder left intact for debugging
)

echo.
pause
