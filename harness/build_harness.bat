@echo off
REM Build the Phase 1b harness. Links against a chosen llama.cpp CUDA build's import libs.
REM Usage: build_harness.bat [build_dir]
setlocal

set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "ROOT=S:\Agent\QSExplore\RecycleLM"
set "SRC=%ROOT%\harness"
set "INC1=%ROOT%\vendor\llama.cpp\include"
set "INC2=%ROOT%\vendor\llama.cpp\ggml\include"
set "LIBDIR=%~1"
if "%LIBDIR%"=="" set "LIBDIR=%ROOT%\build\cuda-graphs"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

echo [build_harness] libs from %LIBDIR%
cl /nologo /O2 /std:c++17 /EHsc /MD /W3 ^
  /I"%INC1%" /I"%INC2%" ^
  "%SRC%\phase1b.cpp" ^
  /Fe:"%SRC%\phase1b.exe" /Fo:"%SRC%\phase1b.obj" ^
  /link "%LIBDIR%\src\llama.lib" "%LIBDIR%\ggml\src\ggml.lib"
if errorlevel 1 ( echo [build_harness] FAILED & exit /b 1 )

echo [build_harness] done: %SRC%\phase1b.exe
