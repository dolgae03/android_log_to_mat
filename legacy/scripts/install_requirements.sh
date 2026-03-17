# Legacy setup script kept for reference.
conda create -n myenv python=3.9.13 -y
source ~/anaconda3/etc/profile.d/conda.sh
conda activate myenv

cd gnss_lib_py
pip install poetry
pip install -r requirements.txt
poetry install

cd ..
pip install -r requirements.txt
mkdir -p ./data
