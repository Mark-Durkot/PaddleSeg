import os
import shutil
import argparse

# Set your directory path here (e.g., your Google Drive path)


def parse_args():
    parser = argparse.ArgumentParser(description='Arrange logs')
    parser.add_argument('--log_directory', type=str, required=True)
    return parser.parse_args()

args = parse_args()
log_directory = args.log_directory
print(f"Log directory: {log_directory}")

def organize_logs(target_dir):
    # Change to the target directory
    if not os.path.exists(target_dir):
        print(f"Error: Path {target_dir} does not exist.")
        return

    os.chdir(target_dir)
    
    # Get all files in the directory
    # We look for files containing 'vdl' or ending in '.log'
    files = [f for f in os.listdir('.') if os.path.isfile(f) and ('vdl' in f or f.endswith('.log'))]
    
    if not files:
        print("No log files found to move.")
        return

    print(f"Found {len(files)} log files. Starting organization...")

    for index, filename in enumerate(sorted(files)):
        # Create a folder name like 'run_01', 'run_02', etc.
        folder_name = f"run_{index + 1:02d}"
        
        # If the folder doesn't exist, create it
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        # Move the file into the new folder
        try:
            shutil.move(filename, os.path.join(folder_name, filename))
            print(f"Moved: {filename} -> {folder_name}/")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

    print("\nOrganization complete! You can now run VisualDL on the parent directory.")

if __name__ == "__main__":
    organize_logs(log_directory)