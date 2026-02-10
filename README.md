# CERBERUS — Autonomous AI Rover

**Operation Ground Truth**

Autonomous intelligence that lives where it works.

---

## What Is Cerberus?

Cerberus (CAAR — Cerberus Autonomous AI Rover) is a 6WD off-road autonomous rover built for real-world outdoor operations in Las Vegas, Nevada. It thinks for itself — all AI inference, decision-making, and logging happens at the edge on a Raspberry Pi 5. No cloud. No remote control. No babysitting.

Cerberus features hot-swappable payload heads that give it different capabilities for different missions: weed scanning, pest deterrence, environmental monitoring, bird identification, surveillance, and microclimate mapping.

## Hardware Platform

- **Chassis:** 6WD Off-Road Robot Car Kit (Seeed Studio) with independent suspension
- **Brain:** Raspberry Pi 5 (8GB) in Argon ONE V3 case
- **Motor Driver:** BTS7960 dual H-bridge (high-current, PWM speed control)
- **Power:** 3S Li-ion battery pack (5000mAh+, BMS protected)
- **Sensors:** INA3221 power monitor, BME680/SCD-40/SHT45 environmental, HC-SR04 ultrasonics
- **Peripherals:** PCA9685 servo controller, NeoPixel status LEDs, MAX98357A I2S audio
- **GPS:** u-blox NEO-6M with external antenna
- **Camera:** Pi Camera 3 with pan-tilt mount
- **OS:** Raspberry Pi OS Lite (Bookworm, 64-bit) — headless

## Payload Heads

| Head | Mission | Key Hardware |
|------|---------|-------------|
| Weed Scanner | Autonomous weed detection + geotag | Camera + GPS + TFLite |
| Surveillance | AI patrol + threat detection | Wide-angle camera + IR LEDs |
| Environmental Logger | Climate data collection | BME680 + SCD-40 sensors |
| Pest Deterrent | Species-aware pest response | PIR + camera + servo decoy + audio |
| Bird Watcher | Neural species identification | Camera + telephoto + BirdNET |
| Microclimate Mapper | GPS-tagged heatmap grids | Extendable arm + SHT45 probe |

## Architecture
```
cerberus/
├── core/           Config, logging, health, safety watchdog
├── mobility/       BTS7960 driver, 6WD skid-steer, GPS navigation
├── comms/          MQTT telemetry, MJPEG streaming
├── perception/     Camera, GPS, environment sensors, obstacle detection
├── intelligence/   TFLite classifier, motion detection, species ID,
│                   A* path planner, behavior tree, adaptive learner
├── heads/          Base class + 6 payload head modules + auto-detection
├── autonomy/       Mission planner, patrol routes, grid driver, RTB
└── storage/        SQLite database (18 tables, 40+ indexes)
```

## Autonomy Principles

- **Edge-first:** All AI inference and decisions run on the Pi — zero cloud dependency
- **Self-navigating:** Plans and executes patrol routes and mission grids autonomously
- **Self-logging:** Every sensor reading, detection, and decision is recorded locally
- **Self-recovering:** Monitors battery, temperature, and system health — returns to base or enters safe mode when needed
- **Adaptive:** Learns from its environment — calibrates confidence thresholds, optimizes patrol zones, predicts pest activity, detects sensor anomalies
- **Safety-first:** Safety watchdog can override any mission and force RTB or shutdown

## Intelligence Layer

Cerberus includes three AI subsystems that work together:

**Behavior Tree** — Reactive decision engine with 7-level priority stack. Emergency stop preempts everything. Safety overrides missions. Obstacle avoidance interrupts navigation. The tree ticks at configurable rates and supports preemptive interrupts with automatic lower-priority task reset.

**A* Path Planner** — Builds a persistent occupancy grid from GPS + obstacle sensor data. Plans paths using A* with octile distance heuristic, 8-directional movement, and Ramer-Douglas-Peucker simplification. Obstacle buffer zones prevent close passes. Grid persists to SQLite across reboots.

**Adaptive Learner** — On-device learning that runs periodically in the background:
- Confidence calibration: auto-adjusts detection thresholds based on verified predictions
- Patrol optimization: identifies high-activity zones and adjusts dwell times
- Weed hotspot prediction: clusters detections and estimates recurrence probability
- Pest activity prediction: builds hourly species activity profiles
- Anomaly detection: sigma-based deviation alerts on sensor baselines

## Development

### Prerequisites

- Python 3.11.9
- VS Code with Python, Pylance, Remote-SSH extensions
- Git

### Setup
```bash
cd C:\Dev\CAAR
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pytest
```

### Running Tests

Full test suite (11 test files, 200+ test cases):
```bash
pytest tests/ -v
```

Individual test modules:
```bash
pytest tests/test_config.py -v
pytest tests/test_db.py -v
pytest tests/test_config_validator.py -v
pytest tests/test_head_detector.py -v
pytest tests/test_obstacle.py -v
pytest tests/test_path_planner.py -v
pytest tests/test_behavior_tree.py -v
pytest tests/test_adaptive_learner.py -v
pytest tests/test_logger.py -v
pytest tests/test_integration_smoke.py -v
```

Standalone smoke test (no pytest required, runs on Pi or dev machine):
```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --config config/cerberus.yaml
python scripts/smoke_test.py --verbose
```

### Test Coverage

| Module | Test File | Focus |
|--------|-----------|-------|
| Config Loader | test_config.py | Singleton, nested access, path resolution, properties |
| Database | test_db.py | 18-table schema, CRUD, transactions, maintenance |
| Config Validator | test_config_validator.py | Types, ranges, GPIO/I2C conflicts, threshold ordering |
| Head Detector | test_head_detector.py | Config override, I2C EEPROM, GPIO combos, fallback |
| Obstacle Detector | test_obstacle.py | Zones, avoidance logic, simulation, polling |
| Path Planner | test_path_planner.py | GPS/grid conversion, A*, simplification, persistence |
| Behavior Tree | test_behavior_tree.py | Composites, decorators, priority, preemption, default tree |
| Adaptive Learner | test_adaptive_learner.py | Calibration, patrol zones, hotspots, pest prediction, anomalies |
| Logger | test_logger.py | File handlers, MQTT handler, format, levels |
| Smoke Test | test_integration_smoke.py | Full boot sequence, cross-system data flow, clean shutdown |

## Project Structure
```
C:\Dev\CAAR\ (local) → /opt/cerberus/ (Pi)
├── cerberus/                    Main Python package (44 files)
├── config/                      YAML configuration
│   ├── cerberus.yaml            Master config
│   └── missions/                Mission definitions
├── models/                      TFLite / ONNX model files
├── data/                        SQLite databases + logs (runtime)
├── scripts/                     Utility and validation scripts
│   ├── install.sh               Dependency installer
│   ├── start.sh                 Launch Cerberus
│   ├── test_motors.py           Hardware validation
│   └── smoke_test.py            Standalone smoke test
├── tests/                       Unit + integration tests (11 files)
├── requirements.txt             Python dependencies
├── cerberus.service             systemd service file
└── .gitignore                   Git exclusions
```

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Foundation | ✅ Complete | Config, SQLite, logger, health monitor, MQTT, safety watchdog |
| 2 — Mobility | ✅ Complete | BTS7960 driver, 6WD skid-steer, motor validation |
| 3 — Perception | ✅ Complete | Camera, GPS, environmental sensors, obstacle detection |
| 4 — Intelligence | ✅ Complete | TFLite classifier, motion detection, species ID |
| 5 — Autonomy | ✅ Complete | Mission planner, waypoint nav, patrol, grid driver, RTB |
| 6 — Heads | ✅ Complete | Base class, 6 head modules, auto-detection |
| 7 — Production Hardening | ✅ Complete | Config validator, path planner, behavior tree, adaptive learner, full test suite |
| 8 — Dashboard | 🔲 Planned | Command dashboard, real-time telemetry, mission control |

## Data Architecture

- **On-Rover (Pi 5):** SQLite — 18 tables, 40+ indexes, WAL mode, zero-infrastructure
- **Dashboard (PC):** SQL Server 2025 — mission history, aggregated telemetry, detection archives
- **Sync:** Cerberus logs locally → syncs to dashboard when connected → dashboard writes to SQL Server

## MQTT Topics
```
cerberus/telemetry/health       Battery, temp, CPU, GPS
cerberus/telemetry/sensors      Environmental readings
cerberus/detections/{type}      AI detections
cerberus/mission/status         Mission state machine
cerberus/command/#              Inbound commands
cerberus/alerts                 Critical alerts
```

## Desert Hardening

Built for Las Vegas conditions:
- Heat dissipation monitoring with multi-tier thermal shutdown
- Dust ingress considerations for all sensors
- UV degradation awareness for exposed components
- Monsoon moisture protection for electronics
- Battery chemistry protection in extreme temperatures

## Builder

Scott — 25+ year software developer, building his first physical robotics platform. Production-quality engineering standards. No compromises.

---

*Cerberus doesn't ask for permission. It asks for forgiveness. Actually, it doesn't do that either. It just logs what it did and moves on.*