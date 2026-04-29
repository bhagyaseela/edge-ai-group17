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

# ─── Train occupancy model with synthetic data ────────────────
def train_occupancy_model():
    # Features: [lux, lux_change, hour]
    # Label: 1 = occupied, 0 = empty
    X = [
        # Occupied patterns (lights on, active changes)
        [300, 20, 9],  [400, 15, 10], [350, 25, 11],
        [500, 30, 12], [450, 20, 14], [400, 18, 15],
        [300, 22, 16], [250, 15, 17], [200, 10, 18],
        [150, 12, 19], [100, 8,  20], [120, 10, 21],
        # Empty patterns (stable low or high lux, no changes)
        [10,  1,  2],  [5,   0,  3],  [8,   1,  4],
        [900, 2,  12], [950, 1,  13], [880, 3,  14],
        [20,  0,  1],  [15,  1,  6],  [800, 2,  7],
        [750, 1,  8],  [30,  2,  23], [10,  0,  0],
    ]
    y = [
        1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0,0,0
    ]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("Occupancy model trained")
    return model

occupancy_model = train_occupancy_model()

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

def get_brightness(lux, scene):
    threshold = learned_thresholds[scene]
    # The brighter it is outside, the less we need indoor lights
    ratio = lux / (threshold + 1e-6)
    brightness = max(0, min(100, int((1 - ratio) * 100)))
    return brightness

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
def predict_occupancy(lux, hour):
    if len(history) < 2:
        return "unknown"
    lux_change = abs(lux - list(history)[-1])
    features   = [[lux, lux_change, hour]]
    prediction = occupancy_model.predict(features)[0]
    return "occupied" if prediction == 1 else "empty"

# ─── Feature 5: Energy Saving Calculator ─────────────────────
def calculate_energy(brightness):
    global total_saved_wh, last_time
    now      = time.time()
    elapsed  = (now - last_time) / 3600   # hours
    last_time = now

    actual_watts = BASELINE_WATTS * (brightness / 100)
    saved_watts  = BASELINE_WATTS - actual_watts
    total_saved_wh += saved_watts * elapsed

    return {
        "actual_watts":    round(actual_watts, 2),
        "saved_watts":     round(saved_watts, 2),
        "total_saved_wh":  round(total_saved_wh, 4),
        "saving_pct":      round((saved_watts / BASELINE_WATTS) * 100, 1)
    }

# ─── MQTT Message Handler ─────────────────────────────────────
def on_message(client, userdata, msg):
    data  = json.loads(msg.payload)
    lux   = data["lux"]
    hour  = data["hour_sim"]
    history.append(lux)

    # Run all 5 AI features
    scene      = detect_scene(lux, hour)
    update_learning(scene, lux)
    brightness = get_brightness(lux, scene)
    is_anomaly, reason = detect_anomaly(lux)
    occupancy  = predict_occupancy(lux, hour)
    energy     = calculate_energy(brightness)

    # Publish lighting command
    command = {
        "brightness_pct": brightness,
        "scene":          scene,
        "occupancy":      occupancy,
        "timestamp":      time.time()
    }
    client.publish(CMD_TOPIC, json.dumps(command))

    # Publish energy data
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