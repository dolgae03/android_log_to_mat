# rebuilt_txt_parser

기존 `test.py` / `src/android.py`는 건드리지 않고, Android GNSS `.txt` 로그를 새로 파싱하는 독립 폴더입니다.

목표는 최종 산출물 포맷을 `test.py`가 쓰는 `Measurement.headers()` 구조에 맞추는 것입니다.

## 구성

- `parser.py`: `Raw` 섹션 추출, 파생 컬럼 계산, `Measurement` 변환, TSV 저장
- `run_sample.py`: 샘플 로그를 바로 돌리기 위한 실행 진입점
- `handoff_schema.py`: 다른 GPT에게 넘길 CSV/TSV 스키마 설명 출력
- `CSV_HANDOFF.md`: 사람이 읽는 handoff 문서

## 기본 실행

프로젝트 루트에서:

```bash
python3 -m rebuilt_txt_parser.parser
```

또는:

```bash
python3 rebuilt_txt_parser/run_sample.py
```

기본 입력은 루트의 `29740_gnss_log_2025_05_07_15_36_12.txt` 이고, 기본 출력은 아래 경로입니다.

```text
rebuilt_txt_parser/output/29740_gnss_log_2025_05_07_15_36_12.tsv
```

## 직접 경로 지정

```bash
python3 -m rebuilt_txt_parser.parser ./some_log.txt -o ./output/result.tsv
```

## 현재 출력 구조

출력 헤더는 `src/data_class.py`의 `Measurement.headers()`를 그대로 사용합니다.

- 시간: `t_sec`, `gps_week`, `tow_sec`
- 위성 메타: `constellation`, `prn`, `frequency_hz`, `code_type`
- 위성 상태: `sv_pos_*`, `sv_vel_*`, `sv_clock_bias`, `sv_clock_drift`
- 측정값: `pseudorange_m`, `phase_cycle`, `doppler_hz`, `snr_dbhz`, `loi`
- 정답 위치: `gt_pos_*`

현재 버전은 위성 위치/속도와 ground truth는 채우지 않고 `NaN`으로 둡니다.

## 참고

- TSV 변환 자체는 `pandas` 없이 동작합니다.
- `parse_txt_to_dataframe()`는 `pandas`가 설치된 환경에서만 사용 가능합니다.

## handoff 스펙 출력

JSON 스펙:

```bash
python3 -m rebuilt_txt_parser.handoff_schema --format json
```

프롬프트 텍스트:

```bash
python3 -m rebuilt_txt_parser.handoff_schema --format prompt
```
