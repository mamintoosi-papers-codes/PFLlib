# راهنمای مقایسه FedAvg و SR-FedAvg
# Comparison Guide: FedAvg vs SR-FedAvg

## فایل‌های موجود / Available Files

### 1. نوت‌بوک ژوپیتر / Jupyter Notebook
📓 **`Compare_FedAvg_SR-FedAvg.ipynb`**
- نوت‌بوک کامل با توضیحات دو زبانه (فارسی/انگلیسی)
- شامل اجرای آزمایش‌ها، تحلیل و نمودارها
- مناسب برای اجرای تعاملی

### 2. اسکریپت پایتون / Python Script
🐍 **`compare_algorithms.py`**
- اسکریپت خودکار برای اجرای مقایسه
- اجرای هر دو الگوریتم و تحلیل نتایج
- تولید نمودارها و جداول

### 3. فایل دستوری ویندوز / Windows Batch File
⚡ **`run_comparison.bat`**
- اجرای سریع با یک کلیک
- تنظیمات قابل ویرایش در ابتدای فایل
- مناسب برای کاربران ویندوز

---

## روش اول: استفاده از نوت‌بوک ژوپیتر / Method 1: Using Jupyter Notebook

### نصب Jupyter (در صورت نیاز) / Install Jupyter (if needed)
```bash
pip install jupyter notebook matplotlib pandas h5py
```

### اجرای نوت‌بوک / Run Notebook
```bash
jupyter notebook Compare_FedAvg_SR-FedAvg.ipynb
```

سپس سلول‌ها را به ترتیب اجرا کنید / Then run cells in order

---

## روش دوم: استفاده از اسکریپت پایتون / Method 2: Using Python Script

### اجرای مستقیم / Direct Execution
```bash
python compare_algorithms.py
```

این اسکریپت به صورت خودکار:
1. داده‌های MNIST را بررسی/تولید می‌کند
2. FedAvg را اجرا می‌کند
3. SR-FedAvg را اجرا می‌کند
4. نتایج را تحلیل و نمودارها را ذخیره می‌کند

This script automatically:
1. Checks/generates MNIST data
2. Runs FedAvg
3. Runs SR-FedAvg
4. Analyzes results and saves plots

---

## روش سوم: استفاده از فایل Batch (ویندوز) / Method 3: Using Batch File (Windows)

### اجرا / Execution
دابل کلیک روی فایل `run_comparison.bat`

یا از کامند لاین:
```cmd
run_comparison.bat
```

### تنظیمات قابل ویرایش / Editable Settings
فایل `run_comparison.bat` را باز کنید و تنظیمات زیر را ویرایش کنید:

```batch
set DATASET=MNIST          # نام دیتاست
set MODEL=CNN              # نوع مدل
set ROUNDS=100             # تعداد دور آموزش
set CLIENTS=20             # تعداد کلاینت‌ها
set JOIN_RATIO=0.5         # نسبت مشارکت در هر دور
set LR=0.01                # نرخ یادگیری
set LOCAL_EPOCHS=5         # تعداد epoch محلی
set BATCH_SIZE=10          # اندازه batch
set DEVICE=cuda            # cuda یا cpu
set TIMES=3                # تعداد تکرار آزمایش
set SR_BETA=0.9            # ضریب momentum برای SR-FedAvg
```

---

## روش چهارم: اجرای دستی / Method 4: Manual Execution

### 1. تولید داده‌ها / Generate Data
```bash
cd dataset
python generate_MNIST.py noniid - balance
cd ..
```

### 2. اجرای FedAvg
```bash
cd system
python main.py -data MNIST -m CNN -algo FedAvg -gr 100 -ls 5 -lr 0.01 -lbs 10 -nc 20 -jr 0.5 -ncl 10 -dev cuda -eg 1 -t 3 -go comparison
cd ..
```

### 3. اجرای SR-FedAvg
```bash
cd system
python main.py -data MNIST -m CNN -algo SR-FedAvg -gr 100 -ls 5 -lr 0.01 -lbs 10 -nc 20 -jr 0.5 -ncl 10 -dev cuda -eg 1 -t 3 -srbeta 0.9 -go comparison
cd ..
```

### 4. تحلیل نتایج / Analyze Results
```bash
python compare_algorithms.py
```

---

## پارامترهای مهم / Important Parameters

| پارامتر | Parameter | توضیح / Description | مقدار پیش‌فرض / Default |
|---------|-----------|---------------------|------------------------|
| `-data` | Dataset | نام دیتاست | MNIST |
| `-m` | Model | نوع مدل | CNN |
| `-algo` | Algorithm | الگوریتم (FedAvg یا SR-FedAvg) | FedAvg |
| `-gr` | Global Rounds | تعداد دورهای آموزش | 100 |
| `-ls` | Local Epochs | تعداد epoch محلی | 5 |
| `-lr` | Learning Rate | نرخ یادگیری | 0.01 |
| `-nc` | Num Clients | تعداد کلاینت‌ها | 20 |
| `-jr` | Join Ratio | نسبت مشارکت | 0.5 |
| `-srbeta` | SR Beta | ضریب momentum (فقط SR-FedAvg) | 0.9 |
| `-dev` | Device | cuda یا cpu | cuda |
| `-t` | Times | تعداد تکرار آزمایش | 3 |

---

## خروجی‌ها / Outputs

بعد از اجرا، فایل‌های زیر تولید می‌شوند:

### نمودارها / Plots
- 📊 `comparison_results.png` - مقایسه دقت و خطا
- 📊 `stability_analysis.png` - تحلیل پایداری (در نوت‌بوک)

### جداول / Tables
- 📄 `comparison_table.csv` - جدول مقایسه آماری

### داده‌های خام / Raw Data
- 📦 `results/MNIST_FedAvg_comparison_*.h5`
- 📦 `results/MNIST_SR-FedAvg_comparison_*.h5`

---

## نمونه نتایج / Sample Results

```
╔════════════════════════════════════════════════════════════╗
║            جدول مقایسه / Comparison Table                 ║
╠════════════════════════════════════════════════════════════╣
║ Metric                    │ FedAvg        │ SR-FedAvg     ║
╟────────────────────────────┼───────────────┼───────────────╢
║ Final Test Accuracy       │ 0.9145±0.0032 │ 0.9267±0.0021 ║
║ Best Test Accuracy        │ 0.9178        │ 0.9289        ║
║ Final Train Loss          │ 0.2345±0.0045 │ 0.2123±0.0028 ║
║ Convergence Round         │ 78            │ 65            ║
╚════════════════════════════════════════════════════════════╝

📊 بهبود / Improvement: +1.21%
```

---

## عیب‌یابی / Troubleshooting

### مشکل CUDA
اگر CUDA در دسترس نیست، پارامتر device را تغییر دهید:
```bash
set DEVICE=cpu
```
یا در اسکریپت پایتون:
```python
CONFIG['device'] = 'cpu'
```

### کمبود حافظه / Out of Memory
اندازه batch را کاهش دهید:
```bash
-lbs 5
```

### خطای import
کتابخانه‌های مورد نیاز را نصب کنید:
```bash
pip install torch torchvision numpy matplotlib pandas h5py
```

---

## اطلاعات بیشتر / More Information

- 📖 مستندات طراحی: `SR_FedAvg_Design.md`
- 💻 کد سرور SR-FedAvg: `system/flcore/servers/serversrfedavg.py`
- 🔧 فایل اصلی: `system/main.py`

---

## سوالات متداول / FAQ

**Q: چگونه تعداد دورها را تغییر دهم؟**
A: پارامتر `-gr` را تغییر دهید. مثال: `-gr 200`

**Q: چگونه دیتاست دیگری استفاده کنم؟**
A: پارامتر `-data` را تغییر دهید. مثال: `-data Cifar10`

**Q: چگونه sr_beta را تنظیم کنم؟**
A: از پارامتر `-srbeta` استفاده کنید. مقادیر معمول: 0.8-0.95

**Q: آیا می‌توان از GPU استفاده کرد؟**
A: بله، با `-dev cuda` (در صورت وجود CUDA)

---

## لایسنس / License
این کد تحت لایسنس پروژه PFLlib منتشر شده است.

---

**نویسنده / Author**: SR-FedAvg Implementation for PFLlib  
**تاریخ / Date**: January 2026
