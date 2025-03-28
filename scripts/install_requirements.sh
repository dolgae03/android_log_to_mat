# 1. conda 환경 생성
conda create -n myenv python=3.9.13 -y
source ~/anaconda3/etc/profile.d/conda.sh 

# 2. 환경 활성화
conda activate myenv

cd gnss_lib_py

# 3. poetry 설치 (conda 환경 안에서)
pip install poetry
pip install -r requirements.txt

# 4. poetry로 의존성 설치 (pyproject.toml 기준)
poetry install

cd ..
pip install -r requirements.txt

# 5. ./data 폴더 생성
mkdir -p ./data
