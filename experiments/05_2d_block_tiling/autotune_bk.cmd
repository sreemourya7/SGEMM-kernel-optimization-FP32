@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if not exist "block2d_tunable.cu" (
  echo ERROR: Save block2d_tunable.cu in this folder first.
  exit /b 1
)
if not exist "tuning_logs" mkdir "tuning_logs"

echo === Kernel 05: BK sweep, BM=64 BN=64 TM=4 TN=4 ===
echo Configurations: BK=8, BK=16, BK=32

for %%K in (8 16 32) do (
  echo.
  echo ==== Compiling BK=%%K ====
  nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Xptxas=-v -DBM_SIZE=64 -DBN_SIZE=64 -DBK_SIZE=%%K block2d_tunable.cu -o block2d_bk%%K.exe > "tuning_logs\bk%%K_compile.txt" 2>&1
  if errorlevel 1 (
    echo ERROR: Compilation failed for BK=%%K. See tuning_logs\bk%%K_compile.txt
    exit /b 1
  )
  findstr /C:"Used " "tuning_logs\bk%%K_compile.txt"

  echo ==== Correctness and 512 benchmark: BK=%%K ====
  block2d_bk%%K.exe 512 5 1 > "tuning_logs\bk%%K_512.txt" 2>&1
  if errorlevel 1 (
    echo ERROR: 512 test failed for BK=%%K. See tuning_logs\bk%%K_512.txt
    exit /b 1
  )
  findstr /C:"Median perf:" /C:"Verification:" "tuning_logs\bk%%K_512.txt"

  echo ==== 4096 benchmark: BK=%%K ====
  block2d_bk%%K.exe 4096 9 0 > "tuning_logs\bk%%K_4096.txt" 2>&1
  if errorlevel 1 (
    echo ERROR: 4096 benchmark failed for BK=%%K. See tuning_logs\bk%%K_4096.txt
    exit /b 1
  )
  findstr /C:"Median time:" /C:"Median perf:" "tuning_logs\bk%%K_4096.txt"
)

echo.
echo ===== SUMMARY =====
for %%K in (8 16 32) do (
  echo BK=%%K
  findstr /C:"Median perf:" "tuning_logs\bk%%K_512.txt"
  findstr /C:"Median perf:" "tuning_logs\bk%%K_4096.txt"
)
echo.
echo All logs saved in tuning_logs.
endlocal
