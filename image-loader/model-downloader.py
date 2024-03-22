import oras.client
import os

namespace = "kubeconna23.azurecr.io"
repository = "phi2-intro-finetuned-demo"
tag = "latest"
directory = "/llm/"

print("------ Oras pull ------")

client = oras.client.OrasClient()
result = client.pull(target=f"{namespace}/{repository}:{tag}", outdir=directory)

# Get a list of all the files in the directory
original_files = os.listdir(directory)

print("------ Building final ------")
# Print the contents of the directory
for file in original_files:
    print(os.path.join(directory, file))

# Set the prefix of the files you want to merge
prefix_01 = "model-00001-of-00002.safetensors"
prefix_02 = "model-00002-of-00002.safetensors"

# Get a list of all the files in the directory that start with the prefix
files_to_merge_01 = [f for f in os.listdir(directory) if f.startswith(prefix_01)]
files_to_merge_02 = [f for f in os.listdir(directory) if f.startswith(prefix_02)]

# Sort the files in ascending order
files_to_merge_01.sort()
files_to_merge_02.sort()

# Open a new file to write the merged data to
with open(directory + "model-00001-of-00002.safetensors", "wb") as outfile:
    # Loop through each file and write its contents to the output file
    for filename in files_to_merge_01:
        with open(directory + filename, "rb") as infile:
            outfile.write(infile.read())

# Delete the original files
for filename in files_to_merge_01:
    os.remove(directory + filename)

with open(directory + "model-00002-of-00002.safetensors", "wb") as outfile:
    # Loop through each file and write its contents to the output file
    for filename in files_to_merge_02:
        with open(directory + filename, "rb") as infile:
            outfile.write(infile.read())

# Delete the original files
for filename in files_to_merge_02:
    os.remove(directory + filename)

# Get a list of all the files in the directory
files = os.listdir(directory)

print("---- Final files after recontruction -----")
# Print the contents of the directory
for file in files:
    print(os.path.join(directory, file))
