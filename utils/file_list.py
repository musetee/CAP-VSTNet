import os
# Example directory to walk
dir_path = r'C:\Users\56991\Projects\Datasets\Task1\pelvis'

# The path where you want to save the .txt file
output_file_path = os.path.join(dir_path,'output.txt')

# Open the output file in write mode with UTF-8 encoding
with open(output_file_path, 'a', encoding='utf-8') as file:
    # Walk through the directory
    for root, _, fnames in sorted(os.walk(dir_path)):
        # Write the root directory path
        file.write(f"Root: {root}\n")
        # Write file names
        for fname in sorted(fnames):
            file.write(f"File: {fname}\n")
        # Optionally, add a separator between different roots
        file.write("-" * 20 + "\n")

print(f"File saved: {output_file_path}")