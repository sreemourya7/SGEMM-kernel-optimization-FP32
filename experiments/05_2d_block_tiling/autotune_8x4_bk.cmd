@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "block2d_shapes.cu" (
    echo ERROR: block2d_shapes.cu is missing from this folder.
    exit /b 1
)
if not exist "shape_bk_logs" mkdir "shape_bk_logs"

echo === Kernel 05: BK sweep for 128x64 C tile, 8x4 per thread ===
echo BM=128 BN=64 TM=8 TN=4 THREADS=256
echo Testing BK=8, 16, 32. Original source files are unchanged.
echo.

call :test 8
if errorlevel 1 exit /b 1
call :test 16
if errorlevel 1 exit /b 1
call :test 32
if errorlevel 1 exit /b 1

echo.
echo ===== FINAL RESULTS: 4096 x 4096 =====
for %%K in (8 16 32) do (
    echo --- BK=%%K ---
    findstr /C:"Median time:" /C:"Median perf:" "shape_bk_logs\8x4_bk%%K_4096.txt"
)
echo.
echo Prior BK=16 result: 1908.9075 GFLOP/s
echo Logs saved in shape_bk_logs.
exit /b 0

:test
set "BK=%~1"
set "NAME=8x4_bk%BK%"
echo ==== Compiling BK=%BK% ====
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas=-v -DBM_SIZE=128 -DBN_SIZE=64 -DBK_SIZE=%BK% -DTM_SIZE=8 -DTN_SIZE=4 block2d_shapes.cu -o "shape_bk_logs\%NAME%.exe" > "shape_bk_logs\%NAME%_compile.txt" 2>&1
if errorlevel 1 (
    echo ERROR: Compile failed. Check shape_bk_logs\%NAME%_compile.txt
    type "shape_bk_logs\%NAME%_compile.txt"
    exit /b 1
)
findstr /C:"Used " /C:"spill " "shape_bk_logs\%NAME%_compile.txt"

echo -- Verify and benchmark 512 x 512 --
"shape_bk_logs\%NAME%.exe" 512 5 1 > "shape_bk_logs\%NAME%_512.txt" 2>&1
if errorlevel 1 (
    echo ERROR: 512 correctness or benchmark failed.
    type "shape_bk_logs\%NAME%_512.txt"
    exit /b 1
)
findstr /C:"Median perf:" /C:"Verification:" "shape_bk_logs\%NAME%_512.txt"

echo -- Benchmark 4096 x 4096, 9 runs --
"shape_bk_logs\%NAME%.exe" 4096 9 0 > "shape_bk_logs\%NAME%_4096.txt" 2>&1
if errorlevel 1 (
    echo ERROR: 4096 benchmark failed.
    type "shape_bk_logs\%NAME%_4096.txt"
    exit /b 1
)
findstr /C:"Median time:" /C:"Median perf:" "shape_bk_logs\%NAME%_4096.txt"
echo.
exit /b 0
