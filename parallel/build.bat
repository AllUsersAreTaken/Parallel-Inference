@echo off
REM Configure + build the parallel-inference package.
setlocal
set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "CMAKE=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_DIR=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "PATH=%NINJA_DIR%;%PATH%"

set "SRC=%~1"
set "BUILD=%~2"
set "LLAMA=%~3"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

"%CMAKE%" -S "%SRC%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DPARALLEL_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DPARALLEL_LLAMA_DIR=%LLAMA%
if errorlevel 1 ( echo [package] configure FAILED & exit /b 1 )

"%CMAKE%" --build "%BUILD%" --config Release -j 16
if errorlevel 1 ( echo [package] build FAILED & exit /b 1 )
echo [package] done
