import tarfile, glob, os

input_directory = "./LCluster"
output_directory = "."

for file_name in glob.glob(os.path.join(input_directory, "*.tar")):
    if os.path.getsize(file_name) == 0:
        print(f"Skipped empty file: {file_name}")
        continue

    try:
        with tarfile.open(file_name) as f:
            f.extractall(output_directory)
            print(f"Extracted: {file_name}")
    except tarfile.ReadError:
        print(f"Failed to read tar file: {file_name}")