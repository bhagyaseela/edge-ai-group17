import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from sklearn.ensemble import RandomForestClassifier

# ─── Configuration ───────────────────────────────────────────
BROKER      = "mqtt"
SUB_TOPIC   = "sensors/group17/lighting/data"
CMD_TOPIC   = "sensors/group17/lighting/command"
ALERT_TOPIC = "alerts/group17/lighting/status"
ENERGY_TOPIC= "sensors/group17/lighting/energy"

# ─── State ───────────────────────────────────────────────────
history = deque(maxlen=30)

# Adaptive learning — stores average lux per scene
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

# Energy tracking
BASELINE_WATTS  = 60.0   # watts if lights were always 100%
total_saved_wh  = 0.0
last_time       = time.time()
start_time      = time.time()

# ─── Train occupancy model with synthetic data ────────────────
def train_occupancy_model():
    # Features: [lux, lux_change, hour]
    # Label: 1 = occupied, 0 = empty
    X = [
        # Occupied — morning activity
        [200, 25, 7],  [250, 30, 8],  [300, 20, 9],
        [350, 35, 10], [400, 28, 11], [380, 22, 12],
        # Occupied — afternoon activity
        [420, 30, 13], [400, 25, 14], [350, 28, 15],
        [300, 32, 16], [250, 20, 17], [200, 18, 18],
        # Occupied — evening with lights on
        [180, 15, 19], [160, 12, 20], [140, 10, 21],
        # Occupied — high lux with active changes (busy room)
        [700, 40, 10], [750, 45, 11], [800, 38, 14],
        [600, 35, 15], [650, 42, 12], [500, 30, 9],
        # Empty — stable night (no one home)
        [5,   0, 0],   [8,   1, 1],   [10,  0, 2],
        [6,   0, 3],   [7,   1, 4],   [5,   0, 5],
        # Empty — stable day (sunlight, no one home)
        [900, 2, 12],  [920, 1, 13],  [880, 3, 14],
        [850, 2, 11],  [910, 1, 10],  [870, 2, 15],
        # Empty — low stable lux (cloudy, no one home)
        [30,  1, 9],   [25,  0, 10],  [20,  1, 11],
        [15,  0, 14],  [18,  1, 15],  [22,  0, 16],
    ]
    y = [
        1,1,1,1,1,1,  # occupied morning
        1,1,1,1,1,1,  # occupied afternoon
        1,1,1,        # occupied evening
        1,1,1,1,1,1,  # occupied high lux active
        0,0,0,0,0,0,  # empty night
        0,0,0,0,0,0,  # empty stable day
        0,0,0,0,0,0,  # empty low stable
    ]
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X, y)
    print("✅ Occupancy model trained")
    return model


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
    # Keep only last 20 samples per scene
    if len(scene_samples[scene]) > 20:
        scene_samples[scene].pop(0)
    # Update learned threshold as average
    learned_thresholds[scene] = np.mean(scene_samples[scene])

# Smoothing buffer to prevent sudden brightness jumps
brightness_history = deque(maxlen=5)

def get_brightness(lux, scene):
    threshold = learned_thresholds[scene]
    ratio = lux / (threshold + 1e-6)
    raw_brightness = max(0, min(100, int((1 - ratio) * 100)))
    
    # Fix 2: Smooth brightness using rolling average
    # This prevents noise from causing sudden jumps
    brightness_history.append(raw_brightness)
    smoothed = int(np.mean(brightness_history))
    return smoothed

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

# ─── Feature 4: Occupancy Prediction ─────────────────────────
# Occupancy detection state
last_lux_values = deque(maxlen=10)
occupancy_confirmed = False
occupancy_counter   = 0

def predict_occupancy(lux, hour):
    global occupancy_confirmed, occupancy_counter
    last_lux_values.append(lux)

    if len(last_lux_values) < 5:
        return "unknown"

    # Calculate how much lux is changing
    lux_std = np.std(list(last_lux_values))

    # Rule 1: Night time (10pm-6am) + very low lux = empty
    if (hour >= 22 or hour < 6) and lux < 80:
        occupancy_confirmed = False
        return "empty"

    # Rule 2: Active lux changes = someone is moving/working
    if lux_std > 15:
        occupancy_counter = min(occupancy_counter + 1, 10)
    else:
        occupancy_counter = max(occupancy_counter - 1, 0)

    # Rule 3: Sustained activity confirms occupancy
    if occupancy_counter >= 4:
        occupancy_confirmed = True
    elif occupancy_counter <= 1:
        occupancy_confirmed = False

    # Rule 4: Daytime with moderate lux = likely occupied
    if 7 <= hour <= 21 and 100 < lux < 850:
        occupancy_confirmed = True

    return "occupied" if occupancy_confirmed else "empty"

# ─── Feature 5: Energy Saving Calculator ─────────────────────
def calculate_energy(brightness):
    global total_saved_wh, last_time
    now      = time.time()
    elapsed  = (now - last_time) / 3600   # convert to hours
    last_time = now

    actual_watts = BASELINE_WATTS * (brightness / 100)
    saved_watts  = BASELINE_WATTS - actual_watts
    total_saved_wh += saved_watts * elapsed

    # Calculate total saving % across entire session
    total_possible_wh = BASELINE_WATTS * (now - start_time) / 3600
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
    data  = json.loads(msg.payload)
    lux   = data["lux"]
    hour  = data["hour_sim"]
    history.append(lux)

    # Run AI features
    scene      = detect_scene(lux, hour)
    update_learning(scene, lux)
    is_anomaly, reason = detect_anomaly(lux)
    occupancy  = predict_occupancy(lux, hour)
    
    # Fix 1: If room is empty, brightness = 0
    if occupancy == "empty":
        brightness = 0
    else:
        brightness = get_brightness(lux, scene)

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
        print(f"[AI] scene={scene}  brightness={brightness}%  occupancy={occupancy}  saved={energy['saving_pct']}%")
        
# ─── Start ────────────────────────────────────────────────────
print("Edge AI started — listening for sensor data...")
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.subscribe(SUB_TOPIC)
client.loop_forever()