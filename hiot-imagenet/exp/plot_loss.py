import json
import matplotlib.pyplot as plt

def load_and_filter_data(filepath, step=10):
    """
    Loads JSON, converts keys to int, sorts by epoch, 
    and filters for epochs divisible by 'step'.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    epochs = []
    losses = []
    
    # Convert keys to integers and sort
    sorted_epochs = sorted([int(k) for k in data.keys()])
    
    for epoch in sorted_epochs:
        # Filter: only keep epochs 0, 10, 20, etc.
        if epoch % step == 0:
            epochs.append(epoch)
            losses.append(data[str(epoch)]['loss']) # Note: JSON keys are strings when loaded
            
    return epochs, losses

# --- Configuration ---
step = 25
file1 = './json_data/base_runtime.out.json' # Replace with your first file name
file2 = './json_data/ot0.1.out.json' # Replace with your second file name
method1_label = 'Baseline (CE Loss)'
method2_label = 'Ours (CE + Tree-OT Loss)'

# --- Load Data ---
# We assume the file exists. If you are testing, you can comment these out
# and use the dummy data generation block below.
try:
    epochs1, losses1 = load_and_filter_data(file1, step= step)
    epochs2, losses2 = load_and_filter_data(file2, step= step)
except FileNotFoundError:
    print("Files not found. Using dummy data for demonstration.")
    # Dummy data generation for demonstration
    epochs1 = list(range(0, 310, 10))
    losses1 = [4.0 - (x * 0.01) + (0.1 if x % 20 == 0 else -0.1) for x in epochs1]
    epochs2 = list(range(0, 310, 10))
    losses2 = [3.8 - (x * 0.012) for x in epochs2]

# --- Plotting ---
plt.figure(figsize=(10, 7))

# Plot Method 1
plt.plot(epochs1, losses1, marker='o', linestyle='-', color='b', label=method1_label, linewidth=1.75)

# Plot Method 2
plt.plot(epochs2, losses2, marker='s', linestyle='--', color='r', label=method2_label, linewidth=1.75)

# Formatting
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)

plt.tick_params(axis='both', which='major', labelsize= 16)
plt.legend(fontsize=18)
plt.grid(True, linestyle=':', alpha=0.6)

# Set x-axis limit slightly wider than data range for visibility
if epochs1 and epochs2:
    max_epoch = max(max(epochs1), max(epochs2))
    plt.xlim(0, max_epoch + 10)

plt.tight_layout()

# Save or Show
plt.savefig(f'./exp/loss_comparison.pdf', dpi=300)
plt.show()