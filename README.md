# android_log_to_mat

`android_log_to_mat`는 안드로이드 로그 데이터를 `.mat` 형식으로 변환하는 도구입니다.  
주로 MATLAB이나 Python에서 로그 데이터를 분석할 수 있도록 돕습니다.

---

## 📦 설치 방법

필요한 패키지들을 설치하려면 아래 스크립트를 실행하세요:

```bash
bash ./scripts/install_requirements.sh
```

---

## 📂 데이터 준비

변환하고자 하는 `.txt` 로그 파일들을 `./data/raw` 디렉토리에 넣습니다.

```bash
# 예시
./data/raw/sample_log.txt
```

---

## 🚀 전체 데이터 처리

로그 파일들을 `.mat` 형식으로 변환하려면 프로젝트 상위 디렉토리에서 아래 명령어를 실행하세요:

```bash
cd ..
bash process_all_data.sh
```

스크립트가 자동으로 `./data/raw` 디렉토리 내의 모든 `.txt` 파일을 처리하여 `.mat` 파일로 저장합니다.

---

## 📁 결과

변환된 `.mat` 파일은 프로젝트 내부의 지정된 출력 폴더에 저장됩니다. 저장 경로는 사용자의 구현에 따라 다를 수 있습니다.

---

## 🛠️ 기타

- Python 3.x 필요
- MATLAB(R) 또는 `.mat` 파일을 읽을 수 있는 Python 환경 필요 (`scipy.io` 등)

---

## 📞 문의

추가 문의 사항이 있다면 언제든지 이슈를 등록해주세요!
