@echo off
REM Build llama.cpp with CUDA (Ninja + MSVC from VS 2022 BuildTools).
REM Usage: build_llama.bat <build_dir> <ON|OFF for GGML_CUDA_GRAPHS> [extra cmake args]
setlocal enabledelayedexpansion

set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "CMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_DIR=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%NINJA_DIR%;%PATH%"

set "SRC=S:\Agent\QSExplore\RecycleLM\vendor\llama.cpp"
set "BUILD=%~1"
set "GRAPHS=%~2"
shift
shift
set "EXTRA="
:collect
if "%~1"=="" goto done
set "EXTRA=!EXTRA! %~1"
shift
goto collect
:done

if "%BUILD%"=="" ( echo usage: build_llama.bat ^<build_dir^> ^<ON^|OFF^> [extra cmake args] & exit /b 2 )
if "%GRAPHS%"=="" set "GRAPHS=ON"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

echo [build_llama] configure BUILD=%BUILD% GRAPHS=%GRAPHS%
"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_GRAPHS=%GRAPHS% ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_SERVER=OFF ^
  -DLLAMA_BUILD_APP=OFF ^
  -DLLAMA_CURL=OFF ^
  %EXTRA%
if errorlevel 1 ( echo [build_llama] configure FAILED & exit /b 1 )

echo [build_llama] build
"%CMAKE%" --build "%BUILD%" --config Release -j 16
if errorlevel 1 ( echo [build_llama] build FAILED & exit /b 1 )

echo [build_llama] done
