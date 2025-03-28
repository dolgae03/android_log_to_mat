#!/bin/bash

# raw 데이터 디렉토리 지정
cd ..

RAW_DIR="./data/raw"

# raw 디렉토리 내 모든 파일 순회
for file in "$RAW_DIR"/*; do
    # 파일명만 추출
    filename=$(basename "$file")

    # python 스크립트 실행
    python3 test.py -n "$filename"
done
