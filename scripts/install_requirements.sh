# 1. conda 환경 생성 (환경 이름은 원하는 대로 설정 가능)
conda create -n myenv python=3.9.13 -y

# 2. 환경 활성화
conda activate myenv

# 3. requirements.txt 파일에 있는 패키지 설치
pip install -r requirements.txt

# 4. ./data 폴더 생성
mkdir -p ./data