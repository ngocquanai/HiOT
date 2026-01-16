import random
from collections import defaultdict
from pathlib import Path
# Sample data: train_mini/02912_Animalia_Chordata_Actinopterygii_Siluriformes_Ictaluridae_Ameiurus_nebulosus/d615f184-8af4-4c60-b9f8-3081c1607644.jpg 2912 313 50
def sample_few_shot(input_file: str, output_file: str, k: int):
    # Dictionary to store lines grouped by class ID
    # Key: Class ID (e.g., "02912"), Value: List of full lines
    class_groups = defaultdict(list)

    # 1. Read and Group
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Based on your format, the class info is the second element (index 1)
            # or can be parsed from the path. Using index 1 is safer.
            parts = line.split()
            if len(parts) >= 2:
                class_id = parts[1]
                class_groups[class_id].append(line)

    sampled_lines = []

    # 2. Shuffle and Select k
    for class_id, lines in class_groups.items():
        # Shuffle in-place to ensure randomness
        random.shuffle(lines)
        
        # Take up to k images (handles cases where a class might have < k)
        sampled_lines.extend(lines[:k])

    # 3. Write to new file
    with open(output_file, 'w') as f:
        for line in sampled_lines:
            f.write(line + '\n')

    print(f"Successfully sampled {len(sampled_lines)} images ({k} per class) to {output_file}")

if __name__ == "__main__":
    # Configuration
    data_name = "inat21_mini_train"
    INPUT_PATH = f"./data/{data_name}.txt"
    K_VALUES = [1, 2, 4, 8, 16]

    for k in K_VALUES:
        output_name = f"./data/few_shot/{data_name}_k{k}.txt"
        sample_few_shot(INPUT_PATH, output_name, k)