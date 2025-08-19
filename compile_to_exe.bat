@echo off
echo Starting PyInstaller compilation...
echo.

REM Change to the directory containing the script
cd /d "%~dp0"

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
echo.

REM Run PyInstaller with the spec file
echo Running PyInstaller...
pyinstaller --clean report_generator.spec

REM Check if compilation was successful
if exist "dist\ReportGenerator.exe" (
    echo.
    echo ===================================
    echo Compilation successful!
    echo Executable created: dist\ReportGenerator.exe
    echo ===================================
    echo.
    echo Opening dist folder...
    explorer dist
) else (
    echo.
    echo ===================================
    echo Compilation failed!
    echo Please check the output above for errors.
    echo ===================================
)

echo.
pause
