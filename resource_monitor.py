#!/usr/bin/env python3
"""
Resource monitoring module for macOS/M3 systems
Captures CPU, Memory, and GPU usage during model inference
"""

import psutil
import subprocess
import threading
import time
import re
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv
import queue

class ResourceMonitor:
    """Monitor system resources including CPU, Memory, and GPU on macOS"""
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize resource monitor
        
        Args:
            sampling_interval: Time between samples in seconds (default 0.1s = 100ms)
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        self.start_time = None
        self.ollama_process = None
        
        # Check if we can use powermetrics (requires sudo)
        self.gpu_monitoring_available = self._check_gpu_monitoring()
        
    def _check_gpu_monitoring(self) -> bool:
        """Check if GPU monitoring via powermetrics is available"""
        try:
            # Check if we're running as root
            if os.getuid() == 0:
                # Already running as root, test powermetrics directly
                cmd = ['powermetrics', '--samplers', 'gpu_power', '-n', '1']
            else:
                # Not root, test with sudo -n
                cmd = ['sudo', '-n', 'powermetrics', '--samplers', 'gpu_power', '-n', '1']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except:
            return False
    
    def _find_ollama_process(self) -> Optional[psutil.Process]:
        """Find the Ollama process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                # Look for ollama in process name or command line
                if 'ollama' in proc.info['name'].lower():
                    return proc
                if proc.info['cmdline'] and any('ollama' in arg.lower() for arg in proc.info['cmdline']):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def _get_cpu_memory_metrics(self) -> Dict[str, float]:
        """Get CPU and memory metrics using psutil"""
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=0),
            'memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
        }
        
        # Get Ollama-specific metrics if process found
        if self.ollama_process:
            try:
                metrics['ollama_cpu_percent'] = self.ollama_process.cpu_percent()
                metrics['ollama_memory_mb'] = self.ollama_process.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.ollama_process = self._find_ollama_process()
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics using powermetrics"""
        if not self.gpu_monitoring_available:
            return {'gpu_percent': 0.0, 'gpu_freq_mhz': 0}
        
        try:
            # Run powermetrics for a single sample
            if os.getuid() == 0:
                # Already running as root
                cmd = ['powermetrics', '--samplers', 'gpu_power', '-n', '1', 
                       '-i', str(int(self.sampling_interval * 1000))]
            else:
                # Not root, use sudo
                cmd = ['sudo', '-n', 'powermetrics', '--samplers', 'gpu_power',
                       '-n', '1', '-i', str(int(self.sampling_interval * 1000))]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            
            if result.returncode != 0:
                return {'gpu_percent': 0.0, 'gpu_freq_mhz': 0}
            
            output = result.stdout
            
            # Parse GPU metrics from output
            metrics = {}
            
            # GPU Active Residency
            gpu_active = re.search(r'GPU Active\s+Residency:\s+(\d+\.?\d*)%', output)
            if gpu_active:
                metrics['gpu_percent'] = float(gpu_active.group(1))
            else:
                metrics['gpu_percent'] = 0.0
            
            # GPU Frequency
            gpu_freq = re.search(r'GPU\s+Frequency:\s+(\d+)\s+MHz', output)
            if gpu_freq:
                metrics['gpu_freq_mhz'] = int(gpu_freq.group(1))
            else:
                metrics['gpu_freq_mhz'] = 0
                
            # Neural Engine if available
            ane_active = re.search(r'ANE\s+Power:\s+(\d+)\s+mW', output)
            if ane_active:
                metrics['ane_power_mw'] = int(ane_active.group(1))
            
            return metrics
            
        except Exception as e:
            print(f"Warning: GPU monitoring failed: {e}")
            return {'gpu_percent': 0.0, 'gpu_freq_mhz': 0}
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        while self.monitoring:
            timestamp = datetime.now()
            elapsed_time = (timestamp - self.start_time).total_seconds()
            
            # Collect metrics
            metrics = {
                'timestamp': timestamp,
                'elapsed_time': elapsed_time
            }
            
            # Add CPU and memory metrics
            metrics.update(self._get_cpu_memory_metrics())
            
            # Add GPU metrics
            metrics.update(self._get_gpu_metrics())
            
            # Add to queue
            self.data_queue.put(metrics)
            
            # Sleep for sampling interval
            time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = datetime.now()
        self.ollama_process = self._find_ollama_process()
        
        # Clear any old data
        while not self.data_queue.empty():
            self.data_queue.get()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"Resource monitoring started (GPU monitoring: {'enabled' if self.gpu_monitoring_available else 'disabled'})")
    
    def stop_monitoring(self) -> List[Dict]:
        """Stop resource monitoring and return collected data"""
        if not self.monitoring:
            return []
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Collect all data from queue
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        
        print(f"Resource monitoring stopped. Collected {len(data)} samples.")
        return data
    
    def save_to_csv(self, data: List[Dict], filename: str, model_name: str, prompt_id: int = 0):
        """Save monitoring data to CSV file"""
        if not data:
            print("No data to save")
            return
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        # Use filename as-is if it's already a full path, otherwise add results/
        if os.path.isabs(filename) or filename.startswith('results/'):
            filepath = filename
        else:
            filepath = os.path.join('results', filename)
        
        # Define CSV columns
        fieldnames = [
            'timestamp', 'elapsed_time', 'cpu_percent', 'gpu_percent', 
            'memory_mb', 'memory_percent', 'memory_available_mb',
            'gpu_freq_mhz', 'ollama_cpu_percent', 'ollama_memory_mb',
            'model', 'prompt_id'
        ]
        
        # Add optional fields if present
        if data and 'ane_power_mw' in data[0]:
            fieldnames.insert(fieldnames.index('gpu_freq_mhz') + 1, 'ane_power_mw')
        
        # Write to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                # Add model and prompt_id
                row['model'] = model_name
                row['prompt_id'] = prompt_id
                
                # Format timestamp
                if hasattr(row['timestamp'], 'isoformat'):
                    row['timestamp'] = row['timestamp'].isoformat()
                # If it's already a string, leave it as is
                
                # Ensure all fields have default values
                for field in fieldnames:
                    if field not in row:
                        row[field] = 0
                
                writer.writerow({k: row.get(k, '') for k in fieldnames})
        
        print(f"Saved {len(data)} samples to {filepath}")
    
    def get_summary_stats(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate summary statistics from monitoring data"""
        if not data:
            return {}
        
        # Extract numeric columns
        cpu_values = [d.get('cpu_percent', 0) for d in data]
        gpu_values = [d.get('gpu_percent', 0) for d in data]
        memory_values = [d.get('memory_mb', 0) for d in data]
        memory_percent_values = [d.get('memory_percent', 0) for d in data]
        
        ollama_cpu = [d.get('ollama_cpu_percent', 0) for d in data if d.get('ollama_cpu_percent', 0) > 0]
        ollama_mem = [d.get('ollama_memory_mb', 0) for d in data if d.get('ollama_memory_mb', 0) > 0]
        
        summary = {
            'duration_seconds': data[-1]['elapsed_time'] if data else 0,
            'samples_collected': len(data),
            'cpu_avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'cpu_peak': max(cpu_values) if cpu_values else 0,
            'gpu_avg': sum(gpu_values) / len(gpu_values) if gpu_values else 0,
            'gpu_peak': max(gpu_values) if gpu_values else 0,
            'memory_avg_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_peak_mb': max(memory_values) if memory_values else 0,
            'memory_avg_percent': sum(memory_percent_values) / len(memory_percent_values) if memory_percent_values else 0,
            'memory_peak_percent': max(memory_percent_values) if memory_percent_values else 0,
        }
        
        if ollama_cpu:
            summary['ollama_cpu_avg'] = sum(ollama_cpu) / len(ollama_cpu)
            summary['ollama_cpu_peak'] = max(ollama_cpu)
        
        if ollama_mem:
            summary['ollama_memory_avg_mb'] = sum(ollama_mem) / len(ollama_mem)
            summary['ollama_memory_peak_mb'] = max(ollama_mem)
            summary['ollama_memory_delta_mb'] = max(ollama_mem) - min(ollama_mem)
        
        return summary


def enable_gpu_monitoring():
    """
    Helper function to enable GPU monitoring by setting up sudo permissions.
    This needs to be run once before using GPU monitoring.
    """
    print("To enable GPU monitoring, you need to allow powermetrics to run with sudo.")
    print("You can either:")
    print("1. Run this script with sudo: sudo python test_deepseek_resource_impact.py")
    print("2. Add NOPASSWD entry to sudoers for powermetrics (more complex)")
    print("\nFor now, GPU monitoring will be disabled if sudo is not available.")


if __name__ == "__main__":
    # Test the resource monitor
    print("Testing resource monitor...")
    monitor = ResourceMonitor(sampling_interval=0.5)
    
    print("\nStarting 5-second test...")
    monitor.start_monitoring()
    time.sleep(5)
    data = monitor.stop_monitoring()
    
    if data:
        print(f"\nCollected {len(data)} samples")
        summary = monitor.get_summary_stats(data)
        print("\nSummary statistics:")
        for key, value in summary.items():
            print(f"  {key}: {value:.2f}")
        
        # Save test data
        monitor.save_to_csv(data, "resource_monitor_test.csv", "test_model", 0)
    else:
        print("No data collected")