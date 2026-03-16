from datetime import datetime, timezone, timedelta

# ---- 입력 ----
t1 = 1378061795             # seconds
t2 = 1694027531000 / 1000   # milliseconds → seconds

# ---- UTC 변환 ----
dt1_utc = datetime.fromtimestamp(t1, tz=timezone.utc)
dt2_utc = datetime.fromtimestamp(t2, tz=timezone.utc)
diff_seconds = t2 - t1

# ---- GPS epoch (1980-01-06 00:00:00 UTC) ----
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)

# ---- UTC → GPS 변환 ----
# 2025년 기준 GPS와 UTC의 차이는 19초 (2025년 10월 현재까지 19 leap seconds)
LEAP_SECONDS = 19

dt1_gps = dt1_utc - GPS_EPOCH + timedelta(seconds=LEAP_SECONDS)
dt2_gps = dt2_utc - GPS_EPOCH + timedelta(seconds=LEAP_SECONDS)
dt1_utc_gps = dt1_utc + timedelta(seconds=LEAP_SECONDS)
dt2_utc_gps = dt2_utc + timedelta(seconds=LEAP_SECONDS)

# ---- 출력 ----
print("UTC 1:", dt1_utc)
print("UTC 2:", dt2_utc)
print("차이(초):", diff_seconds)
print()
print("GPS 1:", dt1_gps)
print("GPS 2:", dt2_gps)