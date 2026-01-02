@echo off
REM ============================================================
REM اسکریپت مقایسه FedAvg و SR-FedAvg
REM Comparison Script for FedAvg and SR-FedAvg
REM ============================================================

echo.
echo ============================================================
echo مقایسه FedAvg و SR-FedAvg / Comparing FedAvg and SR-FedAvg
echo ============================================================
echo.

REM تنظیمات / Configuration
set DATASET=MNIST
set MODEL=CNN
set ROUNDS=100
set CLIENTS=20
set JOIN_RATIO=0.5
set LR=0.01
set LOCAL_EPOCHS=5
set BATCH_SIZE=10
set DEVICE=cuda
set TIMES=3
set SR_BETA=0.9

echo تنظیمات / Configuration:
echo   Dataset: %DATASET%
echo   Model: %MODEL%
echo   Global Rounds: %ROUNDS%
echo   Clients: %CLIENTS%
echo   Join Ratio: %JOIN_RATIO%
echo   Learning Rate: %LR%
echo   SR Beta: %SR_BETA%
echo.

REM 1. بررسی داده‌ها / Check data
if not exist "dataset\%DATASET%" (
    echo [1/3] تولید داده‌ها / Generating data...
    cd dataset
    python generate_%DATASET%.py noniid - balance
    cd ..
) else (
    echo [1/3] داده‌ها موجود است / Data exists
)

echo.
echo ============================================================
echo [2/3] اجرای FedAvg / Running FedAvg
echo ============================================================
echo.

cd system
python main.py ^
    -data %DATASET% ^
    -m %MODEL% ^
    -algo FedAvg ^
    -gr %ROUNDS% ^
    -ls %LOCAL_EPOCHS% ^
    -lr %LR% ^
    -lbs %BATCH_SIZE% ^
    -nc %CLIENTS% ^
    -jr %JOIN_RATIO% ^
    -ncl 10 ^
    -dev %DEVICE% ^
    -eg 1 ^
    -t %TIMES% ^
    -go comparison

cd ..

echo.
echo ============================================================
echo [3/3] اجرای SR-FedAvg / Running SR-FedAvg
echo ============================================================
echo.

cd system
python main.py ^
    -data %DATASET% ^
    -m %MODEL% ^
    -algo SR-FedAvg ^
    -gr %ROUNDS% ^
    -ls %LOCAL_EPOCHS% ^
    -lr %LR% ^
    -lbs %BATCH_SIZE% ^
    -nc %CLIENTS% ^
    -jr %JOIN_RATIO% ^
    -ncl 10 ^
    -dev %DEVICE% ^
    -eg 1 ^
    -t %TIMES% ^
    -srbeta %SR_BETA% ^
    -go comparison

cd ..

echo.
echo ============================================================
echo تحلیل نتایج / Analyzing results
echo ============================================================
echo.

REM اجرای اسکریپت مقایسه / Run comparison script
python compare_algorithms.py

echo.
echo ============================================================
echo اتمام / Completed
echo ============================================================
echo.
pause
