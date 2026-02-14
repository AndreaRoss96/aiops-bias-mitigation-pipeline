#!/bin/bash

# 1. Initialize Virtual Environment
echo "Creating virtual environment..."
python3 -m venv env
source env/bin/activate

# 2. Install Core Dependencies & AIF360 Extras
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create Local Data Folders
echo "Setting up local data storage..."
mkdir -p env/lib/python3.12/site-packages/aif360/data/raw/adult
mkdir -p env/lib/python3.12/site-packages/aif360/data/raw/compas

# 4. Download Adult Dataset to Project Folder
echo "Downloading Adult dataset..."
curl -s -o env/lib/python3.12/site-packages/aif360/data/raw/adult/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -s -o env/lib/python3.12/site-packages/aif360/data/raw/adult/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
curl -s -o env/lib/python3.12/site-packages/aif360/data/raw/adult/adult.names https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

# 5. The "Professional Bridge": Symlink to Site-Packages
# This tricks AIF360 into looking at your project data folder
echo "Linking project data to AIF360 library..."
AIF_DATA_DIR=$(python -c "import aif360; import os; print(os.path.join(os.path.dirname(aif360.__file__), 'data', 'raw'))")

# Create the library path if it doesn't exist
mkdir -p "$AIF_DATA_DIR"

# Link your project data folder to the library's expected path
ln -sfn "$(pwd)/data/raw/adult" "$AIF_DATA_DIR/adult"

echo "----------------------------------------"
echo "✓ Environment setup complete!"
echo "✓ Data is stored in: $(pwd)/data/raw/adult"
echo "✓ AIF360 is linked to this data."
echo "Run 'source env/bin/activate' to begin."