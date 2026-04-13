import argparse
import csv
import glob
import os
import re

import numpy as np


def parse_folder_name(folder_name):
    dp = re.search(r'dp(\d+)', folder_name)
    tp = re.search(r'tp(\d+)', folder_name)
    pp = re.search(r'pp(\d+)', folder_name)
    mbs = re.search(r'mbs(\d+)', folder_name)
    ga = re.search(r'ga(\d+)', folder_name)
    sl = re.search(r'sl(\d+)', folder_name)
    
    return {
        'dp': int(dp.group(1)) if dp else None,
        'tp': int(tp.group(1)) if tp else None,
        'pp': int(pp.group(1)) if pp else None,
        'micro_batch_size': int(mbs.group(1)) if mbs else None,
        'grad_acc': int(ga.group(1)) if ga else None,
        'seq_len': int(sl.group(1)) if sl else None
    }

def from_readable_format(formatted_str):
    if not isinstance(formatted_str, str):
        return formatted_str
        
    # Remove any whitespace and convert to upper case for consistency
    formatted_str = formatted_str.strip().upper()
    
    # If it's just a number without suffix, return float
    try:
        return float(formatted_str)
    except ValueError:
        pass
    
    # Define multipliers
    multipliers = {
        'T': 1e12,
        'B': 1e9,
        'M': 1e6,
        'K': 1e3
    }
    
    # Extract number and suffix
    number = float(formatted_str[:-1])
    suffix = formatted_str[-1]
    
    if suffix in multipliers:
        return number * multipliers[suffix]
    else:
        raise ValueError(f"Unknown suffix: {suffix}")

def parse_log_line(line):
    tokens_s_gpu_match = re.search(r'Tokens/s/GPU:\s*([\d.]+[KMBT]?)', line)
    mfu_match = re.search(r'MFU:\s+(\d+\.\d+)%', line)

    mfu_value, tokens_value = None, None

    if mfu_match:
        mfu_value = mfu_match.group(1)
        mfu_value = float(mfu_value)
    
    if tokens_s_gpu_match:
        tokens_value = tokens_s_gpu_match.group(1)
    
    return mfu_value, from_readable_format(tokens_value)

def process_file(filepath):
    tokens_s_gpu_values = []
    mfu_values = []
    with open(filepath, 'r') as f:
        for line in f:
            if re.search(r'\[default\d+\]:\[rank \d+\]', line):
                mfu_value, tokens_s_gpu_value = parse_log_line(line)
                if tokens_s_gpu_value is not None:
                    tokens_s_gpu_values.append(tokens_s_gpu_value)
                if mfu_value is not None:
                    mfu_values.append(mfu_value)
    
    #NOTE: skip 3 first beginning (warmup)
    if len(tokens_s_gpu_values) < 3 and len(mfu_values) < 3:
        print(f"Warning: Not enough data points for {filepath}")
        return None, None
    tokens_s_gpu = int(round(np.mean(tokens_s_gpu_values[3:]))) if tokens_s_gpu_values else None
    mfu = int(round(np.mean(mfu_values[3:]))) if mfu_values else None

    return mfu, tokens_s_gpu

def write_csv(data, output_filepath):
    if not data:
        return
    
    fieldnames = ['run_name', 'status', 'dp', 'tp', 'pp', 'micro_batch_size', 'grad_acc', 'seq_len', 'avg_tokens_s_gpu', 'avg_mfu']
    with open(output_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)

def read_status(status_file):
    try:
        with open(status_file, 'r') as f:
            return f.read().strip()
    except:
        return None

def create_subdirectory_metrics(input_folder):
    """Create metrics.csv files in each subdirectory"""
    pattern = os.path.join(input_folder, '**/*.out')
    out_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(out_files)} .out files")
    
    processed_dirs = []
    for file_path in out_files:
        dir_path = os.path.dirname(file_path)
        dir_name = os.path.basename(dir_path)
        output_csv = os.path.join(dir_path, 'metrics.csv')
        
        params = parse_folder_name(dir_name)
        print(f"Processing {file_path}...")
        avg_mfu, avg_tokens_s_gpu = process_file(file_path)
        status = read_status(os.path.join(dir_path, 'status.txt'))
        
        params['run_name'] = dir_name
        write_csv(params, output_csv)

        if status is not None:
            params['status'] = status
            write_csv(params, output_csv)

        if avg_tokens_s_gpu is not None:
            params['avg_tokens_s_gpu'] = avg_tokens_s_gpu
            write_csv(params, output_csv)
            processed_dirs.append(dir_path)
            print(f"Processed {file_path} -> Created metrics.csv")
        
        if avg_mfu is not None:
            params['avg_mfu'] = avg_mfu
            write_csv(params, output_csv)
            processed_dirs.append(dir_path)
            print(f"Processed {file_path} -> Created metrics.csv")
    
    return processed_dirs

def aggregate_metrics(input_folder):
    """Create global_metrics.csv from all subdirectory metrics"""
    top_level_dir = glob.glob(input_folder + '/*')

    for top_dir_path in top_level_dir:
        subdirs = glob.glob(top_dir_path + '/*')    

        aggregated_data = []

        for subdir_path in subdirs:
            metrics_file = os.path.join(subdir_path, 'metrics.csv')
            status_file = os.path.join(subdir_path, 'status.txt')

            folder_name = os.path.basename(subdir_path)

            data = {
                'run_name': folder_name,
                'status': read_status(status_file),
                **parse_folder_name(folder_name)  # Unpack the parsed parameters
            }

            # If metrics.csv exists, read the avg_tokens_s_gpu from it
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        reader = csv.DictReader(f)
                        metrics_data = next(reader)
                        data['avg_tokens_s_gpu'] = int(metrics_data['avg_tokens_s_gpu'])
                        data['avg_mfu'] = int(metrics_data['avg_mfu'])
                except:
                    data['avg_tokens_s_gpu'] = -1
                    data['avg_mfu'] = -1
            else:
                data['avg_tokens_s_gpu'] = -1
                data['avg_mfu'] = -1

            aggregated_data.append(data)
    
        # Write global metrics file
        output_file = os.path.join(top_dir_path, 'global_metrics.csv')
        fieldnames = ['run_name', 'status', 'dp', 'tp', 'pp', 'micro_batch_size', 
                    'grad_acc', 'seq_len', 'avg_tokens_s_gpu', 'avg_mfu']
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_data)
            
        print(f"Created global_metrics.csv with {len(aggregated_data)} entries")

def main():
    parser = argparse.ArgumentParser(description='Process log files and create metrics CSVs')
    parser.add_argument('input_folder', help='Path to the top-level folder containing experiment subfolders')
    args = parser.parse_args()

    # Step 1: Create metrics.csv in each subdirectory
    print("Creating individual metrics.csv files...")
    create_subdirectory_metrics(args.input_folder)

    # Step 2: Create global_metrics.csv
    print("\nAggregating metrics into global_metrics.csv...")
    aggregate_metrics(args.input_folder)

if __name__ == "__main__":
    main()