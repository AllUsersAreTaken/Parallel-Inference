@echo off
REM Build one target in an existing llama.cpp build tree (enables tests).
REM Usage: build_target.bat <build_dir> <target>
setlocal

set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "CMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_DIR=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%NINJA_DIR%;%PATH%"

set "SRC=S:\Agent\QSExplore\RecycleLM\vendor\llama.cpp"
set "BUILD=%~1"
set "TARGET=%~2"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

"%CMAKE%" -S "%SRC%" -B "%BUILD%" -DLLAMA_BUILD_TESTS=ON >nul
if errorlevel 1 ( echo configure FAILED & exit /b 1 )

"%CMAKE%" --build "%BUILD%" --target %TARGET% -j 16
if errorlevel 1 ( echo build FAILED & exit /b 1 )
echo [build_target] done: %TARGET%
