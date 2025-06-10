#!/usr/bin/env python3
"""
Script to run Grokfast MA experiments across all model architectures.
Tests grokk_model, gpt2, gpt2_resv, and gpt2_meta with Grokfast MA.
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_experiment(experiment_name, max_steps=3000):
    """Run a single experiment with the given config."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Build the command
    cmd = [
        sys.executable, 
        "scripts/train_grokk.py",
        f"--config-path=../config/experiments",
        f"--config-name={experiment_name}",
        f"train.max_steps={max_steps}"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Experiment {experiment_name} completed successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        # Print last few lines of output for verification
        output_lines = result.stdout.strip().split('\n')
        print(f"📊 Final metrics:")
        for line in output_lines[-5:]:
            if 'val' in line and 'train' in line:
                print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Experiment {experiment_name} failed!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        
        return False

def main():
    """Run all Grokfast MA experiments."""
    print("🧪 Running Grokfast MA Experiments Across All Architectures")
    print("=" * 70)
    
    # Define experiments to run
    experiments = [
        "grokfast_ma_only",      # Original grokk_model
        "gpt2_grokfast_ma",      # GPT2
        "gpt2_resv_grokfast_ma", # GPT2ResV
        "gpt2_meta_grokfast_ma", # GPT2Meta
    ]
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    # Run each experiment
    for experiment in experiments:
        success = run_experiment(experiment, max_steps=3000)
        results[experiment] = success
        
        # Brief pause between experiments
        if experiment != experiments[-1]:  # Don't pause after last experiment
            print(f"\n⏸️  Pausing for 5 seconds before next experiment...")
            time.sleep(5)
    
    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*70}")
    print(f"📋 EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"⏱️  Total duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    print(f"📊 Results:")
    
    successful = 0
    for experiment, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {experiment:25} {status}")
        if success:
            successful += 1
    
    print(f"\n🎯 Overall: {successful}/{len(experiments)} experiments successful")
    
    if successful == len(experiments):
        print("🎉 All experiments completed successfully!")
        return 0
    else:
        print("⚠️  Some experiments failed. Check logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
