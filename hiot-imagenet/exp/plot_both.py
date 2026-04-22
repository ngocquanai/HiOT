import json
import matplotlib.pyplot as plt

def load_and_filter_all_metrics(filepath, step=10):
    """
    Loads JSON, converts keys to int, sorts by epoch, 
    and filters for epochs divisible by 'step' > 30.
    Returns lists for epochs, losses, and accuracies.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    epochs = []
    losses = []
    accuracies = []
    
    # Convert keys to integers and sort
    sorted_epochs = sorted([int(k) for k in data.keys()])
    
    for epoch in sorted_epochs:
        # Filter: only keep epochs 0, 10, 20, etc. AND epoch > 30
        if epoch % step == 0 :
            epochs.append(epoch)
            losses.append(data[str(epoch)]['loss'])
            accuracies.append(data[str(epoch)]['acc1'])
            
    return epochs, losses, accuracies

# --- Configuration ---
step = 25


file1 = './json_data/base_runtime.out.json'
file2 = './json_data/ot0.1.out.json'
method1_label = 'Baseline (CE Loss)'
method2_label = 'Ours (CE + Tree-OT Loss)'

# --- Load Data ---
try:
    epochs1, loss1, acc1 = load_and_filter_all_metrics(file1, step= step)
    epochs2, loss2, acc2 = load_and_filter_all_metrics(file2, step= step)
except FileNotFoundError:
    print("Files not found. Please check the file paths.")
    epochs1, loss1, acc1 = [], [], []
    epochs2, loss2, acc2 = [], [], []

# --- Plotting ---
# Create a figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ==========================================
# Subplot 1: Loss Comparison (Left)
# ==========================================
ax1.plot(epochs1, loss1, marker='o', linestyle='-', color='b', label=method1_label, linewidth=2)
ax1.plot(epochs2, loss2, marker='s', linestyle='--', color='r', label=method2_label, linewidth=2)

ax1.set_title('Loss Comparison', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=12)

# Set x-axis limit
if epochs1 and epochs2:
    max_epoch = max(max(epochs1), max(epochs2))
    ax1.set_xlim(30, max_epoch + 10)

# ==========================================
# Subplot 2: Accuracy Comparison (Right)
# ==========================================
ax2.plot(epochs1, acc1, marker='o', linestyle='-', color='b', label=method1_label, linewidth=2)
ax2.plot(epochs2, acc2, marker='s', linestyle='--', color='r', label=method2_label, linewidth=2)

ax2.set_title('Top-1 Accuracy Comparison', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=12, loc='lower right')

# Set x-axis limit
if epochs1 and epochs2:
    ax2.set_xlim(30, max_epoch + 10)

# ==========================================
# Final Layout and Save
# ==========================================
plt.tight_layout()
plt.savefig(f'./exp/loss_acc_comparison_each{step}.pdf', dpi=300)
plt.show()