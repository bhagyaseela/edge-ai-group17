import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

# ─── Configuration ───────────────────────────────────────────
BROKER       = "mqtt"
SUB_TOPIC    = "sensors/group17/lighting/data"
CMD_TOPIC    = "sensors/group17/lighting/command"
ALERT_TOPIC  = "alerts/group17/lighting/status"
ENERGY_TOPIC = "sensors/group17/lighting/energy"

# ─── State ───────────────────────────────────────────────────
history            = deque(maxlen=30)
brightness_history = deque(maxlen=5)   # smoothing buffer

learned_thresholds = {
    "Night":   50,
    "Morning": 200,
    "Day":     600,
    "Evening": 300
}
scene_samples = {
    "Night":   [],
    "Morning": [],
    "Day":     [],
    "Evening": []
}

BASELINE_WATTS = 60.0
total_saved_wh = 0.0
last_time      = time.time()
start_time     = time.time()

# ─── Feature 1: Scene Detection ──────────────────────────────
def detect_scene(lux, hour):
    if hour < 6 or hour >= 22:
        return "Night"
    elif hour < 10:
        return "Morning"
    elif hour < 17:
        return "Day"
    else:
        return "Evening"

# ─── Feature 2: Adaptive Learning ────────────────────────────
def update_learning(scene, lux):
    scene_samples[scene].append(lux)
    if len(scene_samples[scene]) > 20:
        scene_samples[scene].pop(0)
    learned_thresholds[scene] = np.mean(scene_samples[scene])

# ─── Feature 3: Z-score Anomaly Detection ────────────────────
def detect_anomaly(lux):
    if len(history) < 10:
        return False, "normal"
    mean    = np.mean(history)
    std     = np.std(history)
    z_score = abs(lux - mean) / (std + 1e-6)
    if z_score > 3:
        if lux < mean:
            return True, "BLACKOUT (sudden darkness)"
        else:
            return True, "FLASH (sudden brightness)"
    return False, "normal"

# ─── Feature 4: Occupancy ────────────────────────────────────
def get_occupancy(occupied_flag):
    return "occupied" if occupied_flag else "empty"

# ─── Feature 5: Smooth Brightness ────────────────────────────
def smooth_brightness(raw_brightness):
    brightness_history.append(raw_brightness)
    return int(np.mean(brightness_history))

# ─── Feature 6: Energy Saving Calculator ─────────────────────
def calculate_energy(brightness):
    global total_saved_wh, last_time
    now     = time.time()
    elapsed = (now - last_time) / 3600
    last_time = now

    actual_watts   = BASELINE_WATTS * (brightness / 100)
    saved_watts    = BASELINE_WATTS - actual_watts
    total_saved_wh += saved_watts * elapsed

    total_possible_wh  = BASELINE_WATTS * (now - start_time) / 3600
    overall_saving_pct = round((total_saved_wh / (total_possible_wh + 1e-6)) * 100, 1)
    overall_saving_pct = min(100, overall_saving_pct)

    return {
        "actual_watts":       round(actual_watts, 2),
        "saved_watts":        round(saved_watts, 2),
        "total_saved_wh":     round(total_saved_wh, 4),
        "overall_saving_pct": overall_saving_pct
    }

# ─── MQTT Message Handler ─────────────────────────────────────
def on_message(client, userdata, msg):
    data     = json.loads(msg.payload)
    lux      = data["lux"]
    hour     = data["hour_sim"]
    occupied = data["occupied"]
    raw_brightness = data["brightness"]

    history.append(lux)

    # Run AI features
    scene     = detect_scene(lux, hour)
    update_learning(scene, lux)
    is_anomaly, reason = detect_anomaly(lux)
    occupancy = get_occupancy(occupied)

    # Smooth brightness (no sudden jumps from noise)
    brightness = smooth_brightness(raw_brightness)

    energy = calculate_energy(brightness)

    # Publish lighting command
    command = {
        "brightness_pct": brightness,
        "scene":          scene,
        "occupancy":      occupancy,
        "timestamp":      time.time()
    }
    client.publish(CMD_TOPIC, json.dumps(command))
    client.publish(ENERGY_TOPIC, json.dumps(energy))

    # Publish alert if anomaly
    if is_anomaly:
        alert = {
            "status":    "ANOMALY",
            "reason":    reason,
            "lux":       lux,
            "timestamp": time.time()
        }
        client.publish(ALERT_TOPIC, json.dumps(alert))
        print(f"[ALERT] {reason} — lux={lux:.1f}")
    else:
        print(f"[AI] scene={scene}  occ={occupancy}  lux={lux:.1f}  brightness={brightness}%  saved={energy['overall_saving_pct']}%")

# ─── Start ────────────────────────────────────────────────────
print("✅ Edge AI started — listening for sensor data...")
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.subscribe(SUB_TOPIC)
client.loop_forever()