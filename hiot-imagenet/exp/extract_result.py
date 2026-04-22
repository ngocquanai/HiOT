import re
import os
import json
def parse_training_log(file_path):
    results = {}
    current_epoch = None

    # Pattern to find the epoch number, e.g., "Epoch: [298]"
    epoch_pattern = re.compile(r"Epoch:\s*\[(\d+)\]")
    
    # Pattern to find the result line, e.g., "* Acc@1 72.154 Acc@5 91.294 loss 1.209"
    # We capture group 1 (Acc@1) and group 2 (loss)
    metrics_pattern = re.compile(r"\*\s+Acc@1\s+([\d\.]+)\s+Acc@5\s+[\d\.]+\s+loss\s+([\d\.]+)")

    with open(file_path, 'r') as f:
        for line in f:
            # 1. Check if the line contains the Epoch number
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                # Update current_epoch (it will repeat for many lines, which is fine)
                current_epoch = int(epoch_match.group(1))
            
            # 2. Check if the line contains the final metrics
            metrics_match = metrics_pattern.search(line)
            if metrics_match and current_epoch is not None:
                acc1 = float(metrics_match.group(1))
                loss = float(metrics_match.group(2))
                
                # Store in dictionary
                results[current_epoch] = {
                    'acc1': acc1,
                    'loss': loss
                }

    return results

root_path = "/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT/hiot-imagenet/sbatch"

file_name = "ot0.1.out"

file_path = os.path.join(root_path, file_name)

file_data = parse_training_log(file_path) 


output_filename = f'./json_data/{file_name}.json'

with open(output_filename, 'w') as f:
    # indent=4 makes the file human-readable (pretty-printed)
    json.dump(file_data, f, indent=4)

print(f"Successfully saved metrics to {output_filename}")
