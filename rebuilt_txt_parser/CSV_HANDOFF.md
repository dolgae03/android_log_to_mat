# CSV Handoff Guide

이 문서는 다른 GPT 또는 외부 처리기에게 `rebuilt_txt_parser` 산출물을 넘길 때 사용하는 설명서입니다.

## 핵심 요약

- 파일은 이름은 CSV처럼 써도 실제 포맷은 `TSV`입니다.
- 구분자는 탭(`\t`)입니다.
- 첫 줄은 헤더입니다.
- 각 행은 `한 epoch + 한 위성 + 한 신호` 측정치입니다.
- `nan`은 아직 채워지지 않은 값입니다.

## 절대 보존해야 하는 컬럼

아래 컬럼은 이미 계산된 값이라 수정하면 안 됩니다.

- `t_sec`
- `gps_week`
- `tow_sec`
- `constellation`
- `prn`
- `frequency_hz`
- `code_type`
- `pseudorange_m`
- `phase_cycle`
- `doppler_hz`
- `snr_dbhz`
- `loi`

## 다른 모델이 채워야 하는 빈 컬럼

아래 컬럼은 현재 placeholder 상태입니다.

- `sv_pos_x`
- `sv_pos_y`
- `sv_pos_z`
- `sv_vel_x`
- `sv_vel_y`
- `sv_vel_z`
- `sv_clock_bias`
- `sv_clock_drift`
- `pr_correction`
- `gt_pos_x`
- `gt_pos_y`
- `gt_pos_z`

채울 수 없으면 `nan`으로 유지해야 합니다.

## constellation 인코딩

- `0`: GPS
- `1`: GALILEO
- `2`: BEIDOU
- `3`: GLONASS
- `4`: QZSS
- `5`: SBAS
- `6`: IRNSS
- `7`: UNKNOWN

## 컬럼 설명

- `t_sec`: GPS epoch 기준 절대 초
- `gps_week`: GPS week
- `tow_sec`: GPS time-of-week 초
- `constellation`: 정수 enum
- `prn`: 위성 PRN
- `sv_pos_*`: 위성 ECEF 위치 [m]
- `sv_vel_*`: 위성 ECEF 속도 [m/s]
- `iono_a*`, `iono_b*`: 현재는 0.0 placeholder
- `sv_clock_bias`: 위성 clock bias [m]
- `sv_clock_drift`: 위성 clock drift [m/s]
- `pr_correction`: 추가 pseudorange correction [m]
- `frequency_hz`: 반송파 주파수 [Hz]
- `code_type`: Android raw의 코드 타입
- `pseudorange_m`: pseudorange [m]
- `phase_cycle`: carrier phase [cycles]
- `doppler_hz`: Doppler [Hz]
- `snr_dbhz`: C/N0 [dB-Hz]
- `loi`: loss-of-lock indicator
- `gt_pos_*`: receiver ground truth ECEF [m]

## 다른 GPT에게 바로 줄 수 있는 지시문

아래 문장을 그대로 넘겨도 됩니다.

```text
You will receive a tab-separated GNSS measurement file.
Each row is one satellite measurement at one GNSS epoch.
Keep the header and delimiter unchanged.
Preserve all existing non-missing values.
Only fill missing values in these columns if you can infer them reliably:
sv_pos_x, sv_pos_y, sv_pos_z,
sv_vel_x, sv_vel_y, sv_vel_z,
sv_clock_bias, sv_clock_drift,
pr_correction,
gt_pos_x, gt_pos_y, gt_pos_z.
If you cannot infer a value reliably, leave it as nan.
Constellation encoding is:
0=GPS, 1=GALILEO, 2=BEIDOU, 3=GLONASS, 4=QZSS, 5=SBAS, 6=IRNSS, 7=UNKNOWN.
Do not modify:
t_sec, gps_week, tow_sec, constellation, prn, frequency_hz, code_type,
pseudorange_m, phase_cycle, doppler_hz, snr_dbhz, loi.
```
