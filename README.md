# GNSS TXT Parser

이 레포의 현재 메인 코드 경로는 `src/gnss_txt_parser/` 입니다.

Android GNSS Logger `.txt` 파일에서 `Raw` 섹션을 파싱하고, 측정치를 `Measurement` 기반 TSV로 변환합니다.

## Main Path

- 메인 패키지: `src/gnss_txt_parser/`
- 메인 출력 포맷: 탭 구분 TSV
- 메인 결과 디렉토리: `data/results/tsv/`
- 레거시 코드: `legacy/`

## Quick Start

샘플 파일 실행:

```bash
PYTHONPATH=./src python3 -m gnss_txt_parser
```

직접 입력/출력 지정:

```bash
PYTHONPATH=./src python3 -m gnss_txt_parser ./some_log.txt -o ./data/results/tsv/result.tsv
```

## Output

출력은 `Measurement.headers()` 순서를 따르는 테이블입니다.

- 이미 채워지는 값: 시간 태그, constellation/prn, signal id, pseudorange, phase, doppler, snr, loi
- 아직 비어 있는 값: 위성 위치/속도, 위성 clock bias/drift, pr correction, ground truth

다른 모델로 후처리할 때는 아래 문서를 참고하면 됩니다.

- 핸드오프 설명: `docs/CSV_HANDOFF.md`
- JSON 스펙 출력: `PYTHONPATH=./src python3 -m gnss_txt_parser.handoff_schema --format json`

## Legacy

이전 변환기와 실험 스크립트는 `legacy/` 로 이동했습니다.
비교, 참고, 회귀 확인용으로만 유지합니다.
