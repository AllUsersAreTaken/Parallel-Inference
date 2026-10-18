@echo off
REM Build one harness .cpp against a llama.cpp CUDA build.
REM Usage: build_tool.bat <build_dir> <src.cpp> <out.exe>
setlocal
set "VSDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "ROOT=S:\Agent\QSExplore\RecycleLM"
set "INC1=%ROOT%\vendor\llama.cpp\include"
set "INC2=%ROOT%\vendor\llama.cpp\ggml\include"
set "LIBDIR=%~1"
set "SRC=%~2"
set "OUT=%~3"

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 failed & exit /b 1 )

cl /nologo /O2 /std:c++17 /EHsc /MD /W3 ^
  /I"%INC1%" /I"%INC2%" ^
  "%SRC%" ^
  /Fe:"%OUT%" /Fo:"%OUT%.obj" ^
  /link "%LIBDIR%\src\llama.lib" "%LIBDIR%\ggml\src\ggml.lib"
if errorlevel 1 ( echo build FAILED & exit /b 1 )
echo [build_tool] done: %OUT%
