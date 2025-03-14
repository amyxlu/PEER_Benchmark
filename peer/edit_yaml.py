import os
import shutil

# Directory containing the YAML files to be copied
source_dir = "kyytc8i9"

# List all directories in the current directory
directories = [d for d in os.listdir(".") if os.path.isdir(d)]

# Iterate over each directory
for directory in directories:
    # Skip the source directory
    if directory == source_dir:
        continue

    # Get the list of YAML files in the source directory
    yaml_files = [f for f in os.listdir(source_dir) if f.endswith(".yaml")]

    # Copy each YAML file to the current directory
    for yaml_file in yaml_files:
        source_path = os.path.join(source_dir, yaml_file)
        destination_path = os.path.join(directory, yaml_file)
        shutil.copy2(source_path, destination_path)

print("YAML files copied successfully.")





# import os
# import yaml

# # Directory containing the YAML files
# directory = "."

# # Loop through all files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".yaml"):
#         yaml_file = os.path.join(directory, filename)

#         # Read the YAML file
#         with open(yaml_file, "r") as file:
#             content = yaml.safe_load(file)

#         # Modify the value of mlp_hidden_multiple
#         if "task" in content:
#             if "mlp_hidden_multiple" in content["task"]:
#                 content["task"]["mlp_hidden_multiple"] = 1 

#         # Write the updated content back to the YAML file
#         with open(yaml_file, "w") as file:
#             yaml.dump(content, file)

#         print(f"Updated {filename}")

# print("All YAML files updated successfully.")


import os

# List of directories
directories = [
    "uhg29zk4", "kyytc8i9", "identity", "ich20c3q", "g8e83omk",
    "fbbrfqzk", "8ebs7j9h", "7str7fhl", "13lltqha"
]

# Loop through each directory
for directory in directories:
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            yaml_file = os.path.join(directory, filename)

            # Read the YAML file
            with open(yaml_file, "r") as file:
                lines = file.readlines()

            # Flag to track if the file was modified
            modified = False

            # Iterate over each line in the file
            for i, line in enumerate(lines):
                if line.startswith("    compression_model_id:"):
                    lines[i] = f"    compression_model_id: {directory}\n"
                    modified = True

            # Write the updated content back to the YAML file if modified
            if modified:
                with open(yaml_file, "w") as file:
                    file.writelines(lines)

                print(f"Updated {yaml_file}")

print("All YAML files updated successfully.")