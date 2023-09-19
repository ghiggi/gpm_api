import os


assets_dir_path = "assets"

# Change current working directory to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create the assets directory
if not os.path.exists(assets_dir_path):
    os.makedirs(assets_dir_path)
