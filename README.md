# Smart Lighting Control - CO326 Edge AI Mini Project

## Group 17

## Group Members
- Member 1: E/20/089 Y.H. Edirimanna
- Member 2: E/20/361 Y.H. Senadheera
- Member 3: E/20/366 A.P.B.P. Senavirathna

## Project Description
An Edge AI based smart lighting control system that simulates 
ambient light levels and automatically adjusts lighting brightness 
using AI. The system detects anomalies, predicts occupancy, 
and calculates energy savings in real time.

## System Architecture
Sensor Simulator → Python Edge AI → MQTT Broker → Node-RED → Dashboard

## How to Run
1. Install Docker Desktop
2. Clone this repository
3. Run: docker-compose up --build
4. Open Node-RED: http://localhost:1880
5. Open Dashboard: http://localhost:1880/ui
6. Run Edge AI: docker exec -it python-edge python edge_ai.py

## MQTT Topics Used
|               Topic              |        Purpose       |
|----------------------------------|----------------------|
| sensors/group17/lighting/data    | Raw sensor data      |
| sensors/group17/lighting/command | AI lighting commands |
| sensors/group17/lighting/energy  | Energy saving data   |
| alerts/group17/lighting/status   | Anomaly alerts       |

## AI Features
1. Scene Detection - detects Morning/Day/Evening/Night
2. Adaptive Learning - learns lux thresholds over time
3. Z-score Anomaly Detection - detects blackouts and flashes
4. Occupancy Prediction - Random Forest ML model
5. Energy Saving Calculator - calculates watts and % saved

## Results
[Add your dashboard screenshots here]

## Challenges
- Setting up Docker on Windows
- Fixing Node-RED gauge display issues
- Tuning Z-score threshold for anomaly detection

## Future Improvements
- Integration with real LDR sensor on ESP32
- TensorFlow Lite model for better occupancy prediction
- Mobile notifications for alerts