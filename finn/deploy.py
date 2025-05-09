from shutil import copy, make_archive
from qonnx.core.modelwrapper import ModelWrapper
import shutil, os

version = 45
deploy_dir = f"/home/eveneiha/finn/workspace/fpga/deploy_v{version}"

# Check if folder exists
if os.path.exists(deploy_dir):
    response = input(f"⚠️  Folder {deploy_dir} already exists. Overwrite? (y/n): ").strip().lower()
    if response != "y":
        print("Aborted by user. Deployment not overwritten.")
        exit(0)
    else:
        print("Overwriting deployment folder...")

# Make fresh directory (will not error if exists)
os.makedirs(deploy_dir, exist_ok=True)

# Load model
model = ModelWrapper("/home/eveneiha/finn/workspace/finn/output_dir/tcn_POSTSYNTH.onnx")
model.set_metadata_prop("pynq_deployment_dir", deploy_dir)

# Fetch .bit and .hwh files
bitfile = model.get_metadata_prop("bitfile")
hwh_file = model.get_metadata_prop("hw_handoff")
deploy_files = [bitfile, hwh_file]

for dfile in deploy_files:
    if dfile is not None:
        copy(dfile, deploy_dir)

# Fetch driver files
src = model.get_metadata_prop("pynq_driver_dir")
print(f"📂 Copying driver files from: {src}")
shutil.copytree(src, deploy_dir, dirs_exist_ok=True)
print("✅ Driver files copied.")

# Add test input files to the deploy folder
input_npy = "/home/eveneiha/finn/workspace/ml/data/input.npy"
labels_npy = "/home/eveneiha/finn/workspace/ml/data/labels.npy"
finn_input_npy = "/home/eveneiha/finn/workspace/ml/data/finn_sample.npy"

shutil.copy(input_npy, deploy_dir)
shutil.copy(labels_npy, deploy_dir)
shutil.copy(finn_input_npy, deploy_dir)
print("✅ Added input.npy and labels.npy to deploy folder.")


# Create ZIP outside the folder to avoid recursive zipping
zip_path = f"/home/eveneiha/finn/workspace/fpga/deploy-on-pynq.zip"
make_archive(zip_path.replace('.zip', ''), 'zip', deploy_dir)

# Move into deploy folder
shutil.move(zip_path, os.path.join(deploy_dir, "deploy-on-pynq.zip"))
print(f"✅ FINN build complete! Archive stored at: {deploy_dir}/deploy-on-pynq.zip")
