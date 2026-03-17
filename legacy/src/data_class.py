from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Union
import math
import csv
import io
from datetime import datetime, timezone, timedelta


# ─────────────────────────────────────────
# Constellation
# ─────────────────────────────────────────
class Constellation(Enum):
    GPS = 0
    GALILEO = 1
    BEIDOU = 2
    GLONASS = 3
    QZSS = 4
    SBAS = 5
    IRNSS = 6
    UNKNOWN = 7

# 흔한 표기(문자/숫자/RINEX/Android 코드) 모두 허용
_CONS_ALIAS = {
    # RINEX 문자
    "G": Constellation.GPS,      # GPS
    "E": Constellation.GALILEO,  # Galileo
    "C": Constellation.BEIDOU,   # BeiDou
    "R": Constellation.GLONASS,  # GLONASS
    "J": Constellation.QZSS,     # QZSS
    "S": Constellation.SBAS,     # SBAS
    "I": Constellation.IRNSS,    # IRNSS (NavIC)

    # 풀네임
    "GPS": Constellation.GPS,
    "GALILEO": Constellation.GALILEO,
    "BEIDOU": Constellation.BEIDOU,
    "BDS": Constellation.BEIDOU,
    "GLONASS": Constellation.GLONASS,
    "QZSS": Constellation.QZSS,
    "SBAS": Constellation.SBAS,
    "IRNSS": Constellation.IRNSS,
    "NAVIC": Constellation.IRNSS,

    # Android GnssStatus (참고: GPS=1, SBAS=2, GLONASS=3, QZSS=4, BEIDOU=5, GALILEO=6, IRNSS=7)
    1: Constellation.GPS,
    2: Constellation.SBAS,
    3: Constellation.GLONASS,
    4: Constellation.QZSS,
    5: Constellation.BEIDOU,
    6: Constellation.GALILEO,
    7: Constellation.IRNSS,

    # 혹시 0 기반 코드가 들어오는 경우를 대비 (당신 enum과 동일)
    0: Constellation.GPS,
}

def to_constellation(x: Union[str, int, Constellation, None]) -> int:
    if isinstance(x, Constellation):
        return x.value
    if isinstance(x, str):
        s = x.strip().upper()
        return _CONS_ALIAS.get(s, Constellation.UNKNOWN).value
    if isinstance(x, (int,)):
        return _CONS_ALIAS.get(int(x), Constellation.UNKNOWN).value
    
    return Constellation.UNKNOWN.value


## ─────────────────────────────────────────
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import csv, os

def _leaf_only_unique_keys(d: Dict[str, Any],
                           out: Dict[str, Any],
                           origins: Dict[str, str],
                           path: str = "") -> None:
    """dict를 DFS로 돌며 leaf에서만 현재 키 이름을 사용. 중복 키면 AssertionError."""
    for k, v in d.items():
        here = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _leaf_only_unique_keys(v, out, origins, here)
        else:
            col = k  # 부모 prefix 제거(=현재 키만 사용)
            if col in out:
                raise AssertionError(
                    f"Duplicate leaf key '{col}' at '{here}' (already from '{origins[col]}')."
                )
            out[col] = v
            origins[col] = here

def _dataclass_to_leaf_row(obj: Any) -> Dict[str, Any]:
    """dataclass(또는 dict) → leaf-only unique dict"""
    if is_dataclass(obj):
        d = asdict(obj)
    elif isinstance(obj, dict):
        d = obj
    else:
        d = dict(getattr(obj, "__dict__", {}))
    out, origins = {}, {}
    _leaf_only_unique_keys(d, out, origins)
    return out

def _pad3(seq, fill=math.nan):
    """길이 3 보장용 보조 함수"""
    seq = list(seq) if seq is not None else []
    return (seq + [fill, fill, fill])[:3]

def _pad4(seq, fill=math.nan):
    seq = list(seq) if seq is not None else []
    return (seq + [fill, fill, fill, fill])[:4]

# ─────────────────────────────────────────
# TimeTag
# ─────────────────────────────────────────
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
SECONDS_IN_WEEK = 7 * 24 * 3600

@dataclass
class TimeTag:
    t_sec: int = math.nan          # 절대초(UTC 기준으로 가정; 필요 시 시스템 정의에 맞게 조정)
    gps_week: int = -1
    tow_sec: int = math.nan        # Time Of Week [s]

    @staticmethod
    def from_sec(t: float) -> TimeTag:
        """
        GPS epoch(1980-01-06)부터의 절대초 t가 들어왔다고 가정하고,
        gps_week / tow_sec를 계산한다.
        """
        if not isinstance(t, (int, float)) or math.isnan(t):
            return TimeTag(t_sec=math.nan, gps_week=-1, tow_sec=math.nan)

        # epoch + t초의 절대시간
        dt = GPS_EPOCH + timedelta(seconds=float(t))
        delta = dt - GPS_EPOCH
        total_sec = delta.total_seconds()
        week = int(total_sec // SECONDS_IN_WEEK)
        tow = float(total_sec - week * SECONDS_IN_WEEK)
        return TimeTag(t_sec=int(t), gps_week=week, tow_sec=int(tow))


# ─────────────────────────────────────────
# SatelliteInfo
# ─────────────────────────────────────────
@dataclass
class SatelliteInfo:
    sv_pos: List[float] = field(default_factory=lambda: [math.nan, math.nan, math.nan])
    sv_vel: List[float] = field(default_factory=lambda: [math.nan, math.nan, math.nan])
    sv_clock_bias: float = math.nan
    sv_clock_drift: float = math.nan
    pr_correction: float = math.nan

    iono_coff: List[float] = field(default_factory=lambda: [0.0]*8)  # 4개 계수
    constellation: Constellation = Constellation.UNKNOWN
    prn: int = -1
    frequency_hz: float = math.nan
    code_type: str = ""

    def validate(self) -> None:
        if len(self.sv_pos) != 3 or len(self.sv_vel) != 3:
            raise ValueError("sv_pos/sv_vel must be length-3 lists.")
        if self.prn < -1:
            raise ValueError("prn must be >= -1.")
        # frequency_hz, code_type에 대한 추가 도메인 검증 필요시 추가


# ─────────────────────────────────────────
# MeasurementValue
# ─────────────────────────────────────────
@dataclass
class MeasurementValue:
    pseudorange_m: float = math.nan    # [m]
    phase_cycle: float = math.nan      # [cycles]
    doppler_hz: float = math.nan       # [Hz]
    snr_dbhz: float = math.nan         # [dB-Hz]
    loi: bool = False                  # loss-of-lock indicator

    def validate_basic(self) -> None:
        # 예시 검증: SNR이 너무 이상하면 에러
        if not math.isnan(self.snr_dbhz) and not (0.0 <= self.snr_dbhz <= 70.0):
            raise ValueError(f"SNR out of plausible range: {self.snr_dbhz}")


# ─────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────
@dataclass
class Measurement:
    time: TimeTag = field(default_factory=TimeTag)
    sat: SatelliteInfo = field(default_factory=SatelliteInfo)
    value: MeasurementValue = field(default_factory=MeasurementValue)
    ground_truth: List[float] = field(default_factory=lambda: [math.nan, math.nan, math.nan])  # [x, y, z] in meters

    def validate(self) -> None:
        self.sat.validate()
        self.value.validate_basic()
        # time에 대한 간단 검증 (원하면 더 강하게)
        if self.time.gps_week < -1:
            raise ValueError("gps_week must be >= -1.")
        
    @staticmethod
    def headers() -> List[str]:
        return [
            "t_sec", "gps_week", "tow_sec",
            "constellation", "prn",
            "sv_pos_x", "sv_pos_y", "sv_pos_z",
            "sv_vel_x", "sv_vel_y", "sv_vel_z",
            "iono_a0", "iono_a1", "iono_a2", "iono_a3",
            "iono_b0", "iono_b1", "iono_b2", "iono_b3",
            "sv_clock_bias", "sv_clock_drift",
            "pr_correction", "frequency_hz", "code_type",
            "pseudorange_m", "phase_cycle", "doppler_hz", "snr_dbhz", "loi",
            "gt_pos_x", "gt_pos_y", "gt_pos_z",
        ]

    def print_header(self) -> None:
        print("\t".join(self.headers()))

    def _to_row_dict(self) -> Dict[str, Any]:
        """
        1) leaf-only + 중복 assert로 평탄화
        2) 리스트 필드들(sv_pos, sv_vel, iono_coff, ground_truth)을 명시적 컬럼으로 확장
        3) bool/None 등 형 변환
        """
        leaf = _dataclass_to_leaf_row(self)

        # --- 리스트/배열 확장 규칙 ---
        # SatelliteInfo가 제공하는 리스트 형태 가정: sv_pos[3], sv_vel[3], iono_coff[4]
        # ground_truth[3]도 x,y,z로 확장 필요 시 여기서 처리(현재 헤더엔 없음이라 확장/무시 선택)
        sv_pos_x, sv_pos_y, sv_pos_z = _pad3(leaf.pop("sv_pos", []))
        sv_vel_x, sv_vel_y, sv_vel_z = _pad3(leaf.pop("sv_vel", []))
        gt_x_m, gt_y_m, gt_z_m = _pad3(leaf.pop("ground_truth", []))
        iono_a0, iono_a1, iono_a2, iono_a3, iono_b0, iono_b1, iono_b2, iono_b3 = leaf.pop("iono_coff", [])

        # 값 보정
        # constellation이 Enum/문자일 수 있음 → 이미 Enum은 _leaf_only_unique_keys에서 name 처리
        # loi는 bool일 수 있음 → int 변환
        if "loi" in leaf and isinstance(leaf["loi"], bool):
            leaf["loi"] = int(leaf["loi"])


        leaf.update({
            "constellation": to_constellation(leaf.get("constellation", None))
        })

        # 평탄 확장 값을 삽입
        leaf.update({
            "sv_pos_x": sv_pos_x, "sv_pos_y": sv_pos_y, "sv_pos_z": sv_pos_z,
            "sv_vel_x": sv_vel_x, "sv_vel_y": sv_vel_y, "sv_vel_z": sv_vel_z,
            "iono_a0": iono_a0, "iono_a1": iono_a1, "iono_a2": iono_a2, "iono_a3": iono_a3,
            "iono_b0": iono_b0, "iono_b1": iono_b1, "iono_b2": iono_b2, "iono_b3": iono_b3,
            "gt_pos_x": gt_x_m, "gt_pos_y": gt_y_m, "gt_pos_z": gt_z_m,
        })

        return leaf

    def to_csv(self, sep: str = "\t") -> str:
        """
        leaf-only 직렬화 + 명시적 리스트 확장 → 고정 헤더 순서로 CSV 한 줄 작성
        """
        row = self._to_row_dict()
        headers = self.headers()

        # 누락/여분 키 검증(개발시 안전장치)
        missing = [h for h in headers if h not in row]
        if missing:
            raise KeyError(f"Missing columns for CSV: {missing}")
        # 여분 키가 있어도 쓰진 않음(필요하면 여기서 경고/예외)
        # extras = [k for k in row.keys() if k not in headers]

        # 순서대로 필드 추출
        fields = [row[h] for h in headers]

        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=sep, lineterminator="")
        writer.writerow(fields)
        return buf.getvalue()


# ─────────────────────────────────────────
# (옵션) 여러 줄 CSV 일괄 파싱/쓰기 헬퍼
# ─────────────────────────────────────────
def measurements_to_csv(meas_list: list[Measurement], sep: str = "\t") -> str:
    return "\n".join(m.to_csv(sep) for m in meas_list)

def measurements_from_csv(text: str, sep: str = "\t") -> list[Measurement]:
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        out.append(Measurement.from_csv(line, sep))
    return out

