import json
import matplotlib.pyplot as plt

def load_and_filter_acc(filepath, step=10):
    """
    Loads JSON, converts keys to int, sorts by epoch, 
    and filters for epochs divisible by 'step' > 30.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    epochs = []
    accuracies = []
    
    # Convert keys to integers and sort
    sorted_epochs = sorted([int(k) for k in data.keys()])
    
    for epoch in sorted_epochs:
        # Filter: only keep epochs 0, 10, 20, etc. AND epoch > 30
        if epoch % step == 0:
            epochs.append(epoch)
            # Fetch 'acc1' instead of 'loss'
            accuracies.append(data[str(epoch)]['acc1']) 
            
    return epochs, accuracies

# --- Configuration ---
step = 25




file1 = './json_data/base_runtime.out.json'
file2 = './json_data/ot0.1.out.json'
method1_label = 'Baseline (CE Loss)'
method2_label = 'Ours (CE + Tree-OT Loss)'

# --- Load Data ---
try:
    epochs1, accs1 = load_and_filter_acc(file1, step= step)
    epochs2, accs2 = load_and_filter_acc(file2, step= step)
except FileNotFoundError:
    print("Files not found. Please check the file paths.")
    epochs1, accs1, epochs2, accs2 = [], [], [], []

# --- Plotting ---
plt.figure(figsize=(10, 7))

# Plot Method 1
plt.plot(epochs1, accs1, marker='o', linestyle='-', color='b', label=method1_label, linewidth=1.75)

# Plot Method 2
plt.plot(epochs2, accs2, marker='s', linestyle='--', color='r', label=method2_label, linewidth=1.75)

# Formatting
# plt.title('Top-1 Accuracy Comparison', fontsize=14)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Top-1 Accuracy (%)', fontsize=18)

plt.tick_params(axis='both', which='major', labelsize= 16)
plt.legend(fontsize=18, loc='lower right') # 'lower right' is usually better for accuracy plots
plt.grid(True, linestyle=':', alpha=0.6)



# Set x-axis limit slightly wider than data range for visibility
if epochs1 and epochs2:
    max_epoch = max(max(epochs1), max(epochs2))
    plt.xlim(30, max_epoch + 10) # Start x-axis from 30 to match your filter

plt.tight_layout()

# Save or Show
plt.savefig(f'./exp/acc_comparison.pdf', dpi=300)
plt.show()