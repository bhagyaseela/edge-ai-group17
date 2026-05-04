import time
import random
import math
import json
import paho.mqtt.client as mqtt

# ─── Configuration ───────────────────────────────────────────
BROKER = "mqtt"
TOPIC  = "sensors/group17/lighting/data"

client = mqtt.Client()
client.connect(BROKER, 1883, 60)

print("✅ Simulator started — publishing light sensor data...")

t = 0

# Occupancy pattern — fixed schedule (seconds per period)
# Each tuple: (occupied, duration_in_readings)
OCCUPANCY_SCHEDULE = [
    (True,  40),   # occupied 40 readings (~2 min)
    (False, 30),   # empty    30 readings (~1.5 min)
    (True,  50),   # occupied 50 readings (~2.5 min)
    (False, 20),   # empty    20 readings (~1 min)
    (True,  35),   # occupied 35 readings (~1.75 min)
    (False, 40),   # empty    40 readings (~2 min)
    (True,  45),   # occupied 45 readings (~2.25 min)
    (False, 25),   # empty    25 readings (~1.25 min)
]

def get_occupancy(t):
    """Returns True/False based on schedule, loops forever"""
    total = sum(d for _, d in OCCUPANCY_SCHEDULE)
    pos   = t % total
    acc   = 0
    for occupied, duration in OCCUPANCY_SCHEDULE:
        acc += duration
        if pos < acc:
            return occupied
    return False

while True:
    # ── Wave 1: Lux (day/night cycle + noise + anomalies) ────
    hour_of_day = (t % 720) / 720 * 24
    base_lux    = 500 + 450 * math.sin((hour_of_day - 6) * math.pi / 12)
    base_lux    = max(0, base_lux)

    # Realistic noise always present
    noise = random.uniform(-30, 30)
    lux   = base_lux + noise

    # Anomaly flags
    is_blackout = False
    is_flash    = False

    # Blackout anomaly
    if random.random() < 0.04:
        lux         = random.uniform(0, 50)
        is_blackout = True

    # Flash anomaly
    if not is_blackout and random.random() < 0.02:
        lux      = random.uniform(900, 1100)
        is_flash = True

    lux = max(0, round(lux, 2))

    # ── Wave 2: Occupancy (independent schedule) ─────────────
    occupied = get_occupancy(t)

    # ── Brightness Logic ─────────────────────────────────────
    # Only calculate brightness if:
    # - Room is occupied
    # - No anomaly happening
    if occupied and not is_blackout and not is_flash:
        # More lux outside = less brightness needed inside
        brightness = max(0, min(100, int((1 - lux / 1000) * 100)))
    else:
        brightness = 0

    # ── Build payload ─────────────────────────────────────────
    payload = {
        "lux":        lux,
        "timestamp":  time.time(),
        "hour_sim":   round(hour_of_day, 1),
        "occupied":   occupied,
        "brightness": brightness
    }

    client.publish(TOPIC, json.dumps(payload))
    print(f"[SIM] lux={lux:.1f}  occupied={occupied}  brightness={brightness}%  hour={hour_of_day:.1f}")

    t += 1
    time.sleep(3)