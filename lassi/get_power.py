# Used for power consumption
import csv
import time
import subprocess
from threading import Thread
import threading
from queue import Queue
import queue
import pandas as pd
import matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlShutdown
from lassi.format_colors import color


def get_gpu_power():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        power = float(result.stdout.strip())
        return power
    except Exception as e:
        print(f"Error collecting GPU power data: {e}")
        return None
            

def power_collection_thread(power_queue, stop_event, handle, interval=0.01):
    """Thread function to collect power data from the GPU."""
    start_time = time.time()
    while not stop_event.is_set():
        try:
            current_time = time.time() - start_time
            # Query GPU power usage
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            # power = lassi.getidlepower.get_gpu_power()
            if power is not None:
                power_queue.put((current_time, power))
            time.sleep(interval)
        except Exception as e:
            print(f"Error querying power: {e}")
            break 
        # time.sleep(0.01)

# Function to read CSV and plot the GPU power usage
def power_collect_data(power_queue, code_source, target_lang, start_exe_time, end_exe_time):
    # Collect power data
    power_data = []
    while not power_queue.empty():
        power_data.append(power_queue.get())
    
    # Write results to CSV
    filename = f"{code_source}_{target_lang}_gpu_power_results.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'GPU Power (W)'])
        for time_elapsed, power in power_data:
            writer.writerow([f"{time_elapsed:.2f}", power])
    
    # print(f"\nMODEL TRIAL: {experiment}")
    # print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"GPU power data has been written to {filename}")
    
    total_energy_joules, ave_power = display_energy_results(filename, start_exe_time, end_exe_time)
    
    return total_energy_joules, ave_power


def plot_gpu_power(file_name, start_time, end_time):
    data = pd.read_csv(file_name)

    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['GPU Power (W)'], label='GPU Power')
    
    # Highlight the inference period
    plt.axvspan(float(start_time), float(end_time), alpha=0.2, color='yellow', label='Execution Period')
    
    plt.axvline(x=float(start_time), color='g', linestyle='--', label='Execution Start')
    plt.axvline(x=float(end_time), color='r', linestyle='--', label='Execution End')
    # plt.axvline(x=float(data['Unload model time'][0]), color='b', linestyle='--', label='Model Unloaded')
    
    plt.title(f'GPU Power Usage Over Time for {file_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('GPU Power (W)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return True  # data['Inference Duration'][0]


# Function to calculate total energy consumption
def calculate_total_energy(file_name, start_time, end_time):
    data = pd.read_csv(file_name)
    
    # Filter data to only include the inference period
    print("Start time: " + str(round(start_time, 3)) + " -- End time: " + str(round(end_time, 3)))
    code_execution_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
    
    if len(code_execution_data) > 1:
        time_interval = code_execution_data['Time'].diff().mean()
    else:
        print("Not enough data points to calculate energy consumption.")
        return 0, 0

    # Calculate total energy consumption in joules
    total_energy = sum(code_execution_data['GPU Power (W)'] * time_interval)
    ave_power = code_execution_data['GPU Power (W)'].mean()

    # Get baseline energy consumption for same duration
    print("\n... Capturing baseline energy consumed for same duration...") 
    baseline_energy = lassi.getidlepower.GetGPUpower.baselinepower(int(end_time - start_time))
    print("... Complete.")
    
    return total_energy, ave_power, baseline_energy


def display_energy_results(file_name, start_time, end_time):
    plot_gpu_power(file_name, start_time, end_time)
    total_energy_joules, ave_power, baseline_energy = calculate_total_energy(file_name, start_time, end_time)
    
    # Convert energy units
    total_energy_wh = total_energy_joules / 3600  # Convert from Joules to Watt-hours
    total_energy_kwh = total_energy_wh / 1000  # Convert from Watt-hours to Kilowatt-hours
    
    execution_duration = end_time - start_time
    print(f"Code execution duration: {execution_duration:.2f} seconds")
    print(f"Total GPU energy consumption for {file_name}:")
    print(f"  {total_energy_kwh:.6f} kWh")
    print(f"  {total_energy_wh:.4f} Wh")
    print(f"  {total_energy_joules:.2f} J")
    print("Average GPU power: " + color.BOLD + f"{ave_power:.2f} W" + color.END)
    print("Total Energy During Exe Runtime: " + color.BOLD + f"  {total_energy_joules:.2f} J" + color.END)
    print("Idle Energy (same duration): " + f"  {baseline_energy:.2f} J")
    print("----------------------------------------------------------------------------------------------------------------------------------------")
    
    return total_energy_joules, ave_power
        
class GetGPUpower:
      
    def idlepower(duration):

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    
        start_power_time = time.time()
        power_queue = queue.Queue()
        stop_power_event = threading.Event()
    
        # Start power collection thread
        power_thread = Thread(target=power_collection_thread, args=(power_queue, stop_power_event, handle))
        power_thread.start()
        # time.sleep(2)
         
        # Continue collecting power data for a short time after inference
        end_power_time = time.time()            
        while (end_power_time - start_power_time) < duration:
            # time.sleep(0.001)
            end_power_time = time.time()
        stop_power_event.set()
        power_thread.join()
        total_energy_joules = power_collect_data(power_queue, "", "")

        # Shutdown NVML
        nvmlShutdown()

        return round(total_energy_joules, 2)
