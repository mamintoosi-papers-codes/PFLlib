#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت مقایسه FedAvg و SR-FedAvg
Comparison Script for FedAvg and SR-FedAvg
"""

import os
import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path

# تنظیمات / Configuration
CONFIG = {
    'dataset': 'MNIST',
    'model': 'CNN',
    'batch_size': 10,
    'learning_rate': 0.01,
    'num_clients': 20,
    'join_ratio': 0.5,
    'global_rounds': 100,
    'local_epochs': 5,
    'num_classes': 10,
    'device': 'cuda',
    'eval_gap': 1,
    'times': 3,
    'sr_beta': 0.9,
}

def run_experiment(algorithm, config):
    """اجرای آزمایش / Run experiment"""
    print(f"\n{'='*60}")
    print(f"شروع آزمایش {algorithm} / Starting {algorithm} experiment")
    print(f"{'='*60}\n")
    
    # ساخت دستور / Build command
    cmd = [
        sys.executable, 'system/main.py',
        '-data', config['dataset'],
        '-m', config['model'],
        '-algo', algorithm,
        '-gr', str(config['global_rounds']),
        '-ls', str(config['local_epochs']),
        '-lr', str(config['learning_rate']),
        '-lbs', str(config['batch_size']),
        '-nc', str(config['num_clients']),
        '-jr', str(config['join_ratio']),
        '-ncl', str(config['num_classes']),
        '-dev', config['device'],
        '-eg', str(config['eval_gap']),
        '-t', str(config['times']),
        '-go', 'comparison',
    ]
    
    # اضافه کردن پارامتر SR-FedAvg / Add SR-FedAvg parameter
    if algorithm == 'SR-FedAvg':
        cmd.extend(['-srbeta', str(config['sr_beta'])])
    
    # اجرای دستور / Execute command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ آزمایش {algorithm} با موفقیت به پایان رسید")
        print(f"✓ {algorithm} experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ خطا در اجرای {algorithm}: {e}")
        print(f"✗ Error running {algorithm}: {e}")
        return False

def load_results(dataset, algorithm, goal, times):
    """خواندن نتایج / Load results"""
    results = {'test_acc': [], 'test_auc': [], 'train_loss': []}
    
    for t in range(times):
        filename = f"results/{dataset}_{algorithm}_{goal}_{t}.h5"
        
        if os.path.exists(filename):
            try:
                with h5py.File(filename, 'r') as f:
                    results['test_acc'].append(np.array(f['rs_test_acc']))
                    results['test_auc'].append(np.array(f['rs_test_auc']))
                    results['train_loss'].append(np.array(f['rs_train_loss']))
                print(f"✓ بارگذاری {filename}")
            except Exception as e:
                print(f"✗ خطا در خواندن {filename}: {e}")
        else:
            print(f"⚠ فایل {filename} یافت نشد")
    
    # محاسبه میانگین / Calculate average
    if len(results['test_acc']) > 0:
        results['test_acc_mean'] = np.mean(results['test_acc'], axis=0)
        results['test_acc_std'] = np.std(results['test_acc'], axis=0)
        results['train_loss_mean'] = np.mean(results['train_loss'], axis=0)
        results['train_loss_std'] = np.std(results['train_loss'], axis=0)
    
    return results

def plot_comparison(fedavg_results, srfedavg_results, save_path='comparison_results.png'):
    """رسم نمودارهای مقایسه / Plot comparison charts"""
    plt.rcParams['figure.figsize'] = (16, 6)
    plt.rcParams['font.size'] = 11
    
    fig, axes = plt.subplots(1, 2)
    
    rounds = range(len(fedavg_results['test_acc_mean']))
    
    # نمودار دقت / Accuracy plot
    ax1 = axes[0]
    ax1.plot(rounds, fedavg_results['test_acc_mean'], 'b-', linewidth=2, label='FedAvg', marker='o', markersize=4, markevery=10)
    ax1.fill_between(rounds, 
                      fedavg_results['test_acc_mean'] - fedavg_results['test_acc_std'],
                      fedavg_results['test_acc_mean'] + fedavg_results['test_acc_std'],
                      alpha=0.2, color='blue')
    
    ax1.plot(rounds, srfedavg_results['test_acc_mean'], 'r-', linewidth=2, label='SR-FedAvg', marker='s', markersize=4, markevery=10)
    ax1.fill_between(rounds,
                      srfedavg_results['test_acc_mean'] - srfedavg_results['test_acc_std'],
                      srfedavg_results['test_acc_mean'] + srfedavg_results['test_acc_std'],
                      alpha=0.2, color='red')
    
    ax1.set_xlabel('Global Rounds', fontsize=13)
    ax1.set_ylabel('Test Accuracy', fontsize=13)
    ax1.set_title('Test Accuracy Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # نمودار خطا / Loss plot
    ax2 = axes[1]
    ax2.plot(rounds, fedavg_results['train_loss_mean'], 'b-', linewidth=2, label='FedAvg', marker='o', markersize=4, markevery=10)
    ax2.fill_between(rounds,
                      fedavg_results['train_loss_mean'] - fedavg_results['train_loss_std'],
                      fedavg_results['train_loss_mean'] + fedavg_results['train_loss_std'],
                      alpha=0.2, color='blue')
    
    ax2.plot(rounds, srfedavg_results['train_loss_mean'], 'r-', linewidth=2, label='SR-FedAvg', marker='s', markersize=4, markevery=10)
    ax2.fill_between(rounds,
                      srfedavg_results['train_loss_mean'] - srfedavg_results['train_loss_std'],
                      srfedavg_results['train_loss_mean'] + srfedavg_results['train_loss_std'],
                      alpha=0.2, color='red')
    
    ax2.set_xlabel('Global Rounds', fontsize=13)
    ax2.set_ylabel('Training Loss', fontsize=13)
    ax2.set_title('Training Loss Comparison', fontsize=15, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ نمودار ذخیره شد: {save_path}")
    plt.show()

def print_comparison_table(fedavg_results, srfedavg_results):
    """چاپ جدول مقایسه / Print comparison table"""
    comparison_data = {
        'Metric': [
            'Final Test Accuracy',
            'Best Test Accuracy',
            'Final Train Loss',
            'Best Train Loss',
            'Convergence Round'
        ],
        'FedAvg': [
            f"{fedavg_results['test_acc_mean'][-1]:.4f} ± {fedavg_results['test_acc_std'][-1]:.4f}",
            f"{np.max(fedavg_results['test_acc_mean']):.4f}",
            f"{fedavg_results['train_loss_mean'][-1]:.4f} ± {fedavg_results['train_loss_std'][-1]:.4f}",
            f"{np.min(fedavg_results['train_loss_mean']):.4f}",
            f"{np.argmax(fedavg_results['test_acc_mean'])}"
        ],
        'SR-FedAvg': [
            f"{srfedavg_results['test_acc_mean'][-1]:.4f} ± {srfedavg_results['test_acc_std'][-1]:.4f}",
            f"{np.max(srfedavg_results['test_acc_mean']):.4f}",
            f"{srfedavg_results['train_loss_mean'][-1]:.4f} ± {srfedavg_results['train_loss_std'][-1]:.4f}",
            f"{np.min(srfedavg_results['train_loss_mean']):.4f}",
            f"{np.argmax(srfedavg_results['test_acc_mean'])}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("جدول مقایسه نتایج / Results Comparison Table")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # محاسبه بهبود / Calculate improvement
    improvement = (
        (np.max(srfedavg_results['test_acc_mean']) - np.max(fedavg_results['test_acc_mean'])) 
        / np.max(fedavg_results['test_acc_mean']) * 100
    )
    
    print(f"\n📊 Improvement: SR-FedAvg vs FedAvg: {improvement:+.2f}%")
    
    # ذخیره / Save
    df.to_csv('comparison_table.csv', index=False)
    print(f"✓ جدول ذخیره شد: comparison_table.csv")
    
    return df

def main():
    """تابع اصلی / Main function"""
    print("\n" + "="*80)
    print("مقایسه FedAvg و SR-FedAvg")
    print("Comparison of FedAvg and SR-FedAvg")
    print("="*80)
    
    print("\nتنظیمات آزمایش / Experiment Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # 1. تولید داده / Generate data
    print("\n[1/4] بررسی داده‌ها / Checking data...")
    if not os.path.exists(f'./dataset/{CONFIG["dataset"]}'):
        print(f"تولید داده‌های {CONFIG['dataset']}...")
        os.chdir('./dataset')
        subprocess.run([sys.executable, f'generate_{CONFIG["dataset"]}.py', 'noniid', '-', 'balance'])
        os.chdir('..')
    else:
        print(f"✓ داده‌های {CONFIG['dataset']} موجود است")
    
    # 2. اجرای FedAvg / Run FedAvg
    print("\n[2/4] اجرای FedAvg / Running FedAvg...")
    run_experiment('FedAvg', CONFIG)
    
    # 3. اجرای SR-FedAvg / Run SR-FedAvg
    print("\n[3/4] اجرای SR-FedAvg / Running SR-FedAvg...")
    run_experiment('SR-FedAvg', CONFIG)
    
    # 4. تحلیل نتایج / Analyze results
    print("\n[4/4] تحلیل نتایج / Analyzing results...")
    
    fedavg_results = load_results(CONFIG['dataset'], 'FedAvg', 'comparison', CONFIG['times'])
    srfedavg_results = load_results(CONFIG['dataset'], 'SR-FedAvg', 'comparison', CONFIG['times'])
    
    if len(fedavg_results['test_acc']) > 0 and len(srfedavg_results['test_acc']) > 0:
        print("\n✓ نتایج با موفقیت بارگذاری شدند")
        
        # نمایش نتایج / Display results
        print(f"\nFedAvg - Best Accuracy: {np.max(fedavg_results['test_acc_mean']):.4f}")
        print(f"SR-FedAvg - Best Accuracy: {np.max(srfedavg_results['test_acc_mean']):.4f}")
        
        # رسم نمودارها / Plot charts
        plot_comparison(fedavg_results, srfedavg_results)
        
        # جدول مقایسه / Comparison table
        print_comparison_table(fedavg_results, srfedavg_results)
        
        print("\n" + "="*80)
        print("✓ مقایسه با موفقیت به پایان رسید!")
        print("✓ Comparison completed successfully!")
        print("="*80)
    else:
        print("\n✗ خطا: نتایج یافت نشد!")
        print("✗ Error: Results not found!")

if __name__ == '__main__':
    main()
