@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "block2d_shapes.cu" (
  echo ERROR: block2d_shapes.cu must be in this folder.
  exit /b 1
)
if not exist "shape_logs" mkdir "shape_logs"

echo === Kernel 05: Shape sweep, 256 threads per block ===
echo Existing block2d_gemm.cu and BK sweep files are not modified.
echo.

call :test 128 64 16 8 4 8x4_bk16
call :test 64 128 16 4 8 4x8_bk16
call :test 128 128 8 8 8 8x8_bk8
call :test 128 128 16 8 8 8x8_bk16

echo.
echo ===== RESULTS at 4096 x 4096 =====
for %%C in (8x4_bk16 4x8_bk16 8x8_bk8 8x8_bk16) do (
  echo --- %%C ---
  if exist "shape_logs\%%C_4096.txt" (
    findstr /C:"Median time:" /C:"Median perf:" "shape_logs\%%C_4096.txt"
  ) else (
    echo Not benchmarked: see compilation and correctness logs.
  )
)
echo.
echo Logs are in shape_logs. Compare against BK32 baseline: 1560.8851 GFLOP/s.
exit /b 0

:test
set "BM=%~1"
set "BN=%~2"
set "BK=%~3"
set "TM=%~4"
set "TN=%~5"
set "NAME=%~6"
echo.
echo ==== %NAME%: A=%BM%x%BK% B=%BK%x%BN% Thread=%TM%x%TN% ====
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas=-v -DBM_SIZE=%BM% -DBN_SIZE=%BN% -DBK_SIZE=%BK% -DTM_SIZE=%TM% -DTN_SIZE=%TN% block2d_shapes.cu -o "shape_logs\%NAME%.exe" > "shape_logs\%NAME%_compile.txt" 2>&1
if errorlevel 1 (
  echo COMPILATION FAILED, see shape_logs\%NAME%_compile.txt
  exit /b 0
)
findstr /C:"Used " /C:"spill " "shape_logs\%NAME%_compile.txt"
echo -- correctness at 512 --
"shape_logs\%NAME%.exe" 512 5 1 > "shape_logs\%NAME%_512.txt" 2>&1
if errorlevel 1 (
  echo VERIFICATION OR LAUNCH FAILED, see shape_logs\%NAME%_512.txt
  exit /b 0
)
findstr /C:"Median perf:" /C:"Verification:" "shape_logs\%NAME%_512.txt"
echo -- benchmark at 4096 --
"shape_logs\%NAME%.exe" 4096 9 0 > "shape_logs\%NAME%_4096.txt" 2>&1
if errorlevel 1 (
  echo BENCHMARK FAILED, see shape_logs\%NAME%_4096.txt
  exit /b 0
)
findstr /C:"Median time:" /C:"Median perf:" "shape_logs\%NAME%_4096.txt"
exit /b 0
