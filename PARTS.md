# CERBERUS — Hardware Parts Checklist

**Operation Ground Truth — Bill of Materials**

Complete hardware inventory for building the Cerberus Autonomous AI Rover. Components are organized by subsystem with procurement status tracking.

> **Budget Note:** Total estimated build cost is $700–$750 across all subsystems and payload heads. The core platform (brain + chassis + power + mobility) runs approximately $400. Payload head sensors and field-hardening add the remainder.

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| ✅ | Ordered / Received |
| 🔲 | Not Yet Ordered |
| ⏳ | Deferred (future round) |

---

## 1. Brain (Compute Stack)

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | Raspberry Pi 5 | 8GB RAM, BCM2712, quad-core 2.4GHz Cortex-A76 | $134 | The brain — all AI runs here |
| ✅ | Argon ONE V3 Case | Aluminum, full HDMI routing, GPIO access, fan | $49 | Thermal management + port routing |
| ✅ | microSD Card | Samsung EVO Plus 128GB, A2/V30 | $14 | Boot drive (NVMe upgrade later) |
| 🔲 | Pi 5 Camera Cable | 22-pin to 15-pin FPC, 200mm | $8 | Required for Pi Camera Module 3 on Pi 5 |
| ⏳ | NVMe SSD | 256GB M.2 2230/2242 | $25 | Future boot drive via Argon V3 M.2 slot |

---

## 2. Chassis & Drivetrain

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | 6WD Off-Road Chassis Kit | Seeed Studio, 2mm aluminum, 6x DC motors, 1:34 gearbox, independent suspension | $100 | [Electromaker](https://www.electromaker.io) — Source: Seeed Studio |
| ✅ | BTS7960 Motor Drivers | 43A dual H-bridge, PWM + direction, x2 | $17 | One per side (left 3 motors / right 3 motors) |

---

## 3. Power System

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | 3S Li-ion Batteries | Zeee 11.1V 5200mAh, XT60, 50C, x2 | $47 | Two batteries — swap one while charging |
| ✅ | LiPo Balance Charger | HTRC T150, 150W, multi-chemistry | $36 | Charges 1S–6S LiPo/Li-ion/NiMH |
| ✅ | 5V 5A Buck Converter | DROK XL4015, fixed output | $10 | Steps 11.1V → 5V for Pi field power |
| ✅ | INA3221 Power Monitor | HiLetgo triple-channel, I2C, x2 | $8 | Monitors battery, 5V rail, motor current |
| ✅ | Fuse Block | RED WOLF 6-way, damp-proof cover | $14 | Organized power distribution |
| ✅ | LiPo Safe Bag | Tenergy fireproof, 7x9" | $9 | **Critical** — Las Vegas heat + lithium = respect |
| ✅ | XT60 Connectors | Male/female pairs + 12AWG silicone wire | ~$8 | Included in battery/charger kit |

---

## 4. Perception (Sensors)

### Camera

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | Pi Camera Module 3 | Sony IMX708, 12MP, PDAF autofocus, HDR, 75° FoV | $28 | Standard lens — all-purpose starting camera |
| ⏳ | Pi Camera Module 3 Wide | 120° FoV variant | $35 | Surveillance head upgrade |
| ⏳ | Pi Camera Module 3 NoIR | No IR filter — night vision with IR LEDs | $25 | Night surveillance variant |
| ⏳ | Telephoto Lens Adapter | Clip-on for Pi Camera | $15–25 | Bird Watcher head — distance ID |

### Navigation & Ranging

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | GPS Module | HiLetgo NEO-6M, ceramic antenna, EEPROM, RS232 TTL | $9 | Waypoint navigation + geotag |
| ✅ | Ultrasonic Sensors | Smraza HC-SR04, 5-pack, includes 2 brackets | $10 | Front obstacle array (3 active + spares) |
| ✅ | Compass Module | GY-271 QMC5883L, 3-axis, I2C, 3–5V | $7 | Heading without GPS drift |

### Environmental

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | BME680 Breakout | Temp / humidity / pressure / VOC gas, I2C + SPI, 3.3V/5V | $17 | Environmental Logger head |
| ✅ | SHT31-D Breakout | Temp / humidity, ±0.3°C, I2C, 3-pack | $14 | Microclimate Mapper probe + spares |
| ✅ | SCD40 CO2 Sensor | NDIR CO2, temp, humidity, I2C, 400–2000 ppm | $24 | Environmental Logger — air quality |

---

## 5. Servo & Control

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | PCA9685 PWM Driver | Teyleten Robot, 16-channel I2C servo controller, 2-pack | $13 | Camera pan-tilt + pest head decoy |
| ✅ | SG90 Micro Servos | 120° rotation, x2 | $7 | Camera pan-tilt mount |
| ⏳ | Pan-Tilt Bracket | Metal, camera mount | $10–15 | Measurement-dependent — buy after chassis assembly |

---

## 6. Payload Head Hardware

### Pest Deterrent Head

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | PIR Motion Sensors | HC-SR501, 5-pack | $9 | Warm body detection trigger |
| ✅ | NeoPixel LED Rings | WS2812B 16-bit, 2-pack | $13 | Predator eye animation |
| ✅ | I2S Audio Amplifier | MAX98357A, 3W Class D, 2-pack | $7 | Predator audio playback |
| ✅ | Mini Speakers | Gikfun 40mm 3W 4Ω, 2-pack | $9 | Pairs with MAX98357A |

### Surveillance Head

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | IR LED Boards | 850nm, 3W, adjustable resistor, 2-pack | $7 | Night vision illumination for Pi Camera |

### Weatherproofing

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | Conformal Coating | Silicone, 100ml | $13 | PCB-level moisture and dust protection |
| ⏳ | Weatherproof Enclosure | IP65/IP67, NEMA rated, vented + fan | $30–50 | Size TBD after chassis assembly |
| ⏳ | Cable Glands | PG7/PG9/PG11 assortment | $10–12 | Sealed wire pass-throughs for enclosure |

---

## 7. Wiring & Assembly

| Status | Item | Specs | Est. Price | Notes |
|--------|------|-------|-----------|-------|
| ✅ | Dupont Jumper Wires | ELEGOO 120pcs, M-F / M-M / F-F | $7 | Prototyping + sensor hookup |
| ✅ | Breadboards | ELEGOO 3-pack, 830-point | $9 | Bench prototyping (not for field use) |
| ✅ | M3 Standoff Kit | Aienxn 120pcs, brass + nylon | $9 | Board mounting to chassis |
| ✅ | Heat Shrink Tubing | DHOOZ 900pcs, marine grade, 12 sizes | $7 | Waterproof joint protection |
| ✅ | Zip Ties + Cable Mounts | 140-pack combo, adhesive + screw mounts | $10 | Cable management on chassis |

---

## 8. Tools (Already Owned)

| Item | Model | Notes |
|------|-------|-------|
| Soldering Station | WEP 927-IV 110W | For header pins + permanent field connections |
| Multimeter | — | Continuity, voltage, resistance checks |
| Dev Machine | Windows PC | Python 3.11.9, VS Code, venv |

---

## Procurement Summary

| Category | Estimated Cost |
|----------|---------------|
| Brain (Compute Stack) | ~$205 |
| Chassis & Drivetrain | ~$117 |
| Power System | ~$132 |
| Perception (Sensors) | ~$109 |
| Servo & Control | ~$20 |
| Payload Head Hardware | ~$58 |
| Wiring & Assembly | ~$42 |
| **Core Platform Total** | **~$683** |
| Deferred Items (enclosure, camera variants, NVMe) | ~$115–165 |
| **Full Build Estimate** | **~$800–850** |

---

## Notes

### Desert Hardening (Las Vegas, NV)
Every component choice accounts for extreme conditions:
- **Heat:** 115°F+ ambient, 130°F+ surface temps — active cooling required, conformal coating on all boards
- **Dust:** Fine desert particulate — sealed or filtered enclosures, cable glands on all pass-throughs
- **UV:** Year-round UV degradation — UV-resistant enclosure materials, protected cable runs
- **Monsoon:** July–September rain bursts — IP65+ weatherproofing for field deployment

### Soldering Required
- INA3221 and PCA9685 ship with unsoldered header pins
- All permanent field connections must be soldered (breadboards fail under vibration)
- Soldering station: WEP 927-IV 110W (owned)

### Power Architecture
- **Battery:** 3S Li-ion 11.1V → BTS7960 motor drivers (direct)
- **Pi 5:** 11.1V → XL4015 buck converter → 5V/5A → USB-C power input
- **Sensors:** 3.3V from Pi 5 GPIO or 5V rail via INA3221 monitoring
- **Fuse Protection:** 6-way fuse block distributes and protects all circuits

### I2C Bus Map (Active Devices)
| Address | Device | Purpose |
|---------|--------|---------|
| 0x40 | PCA9685 | Servo PWM control |
| 0x40–0x42 | INA3221 | Power monitoring (3 channels) |
| 0x44 | SHT31-D | Temperature / humidity |
| 0x61 | SCD40 | CO2 / temperature / humidity |
| 0x0D | QMC5883L | Compass heading |
| 0x76/0x77 | BME680 | Environmental (temp/hum/pres/gas) |

> **I2C Note:** Address conflicts are checked at startup by `config_validator.py`. The PCA9685 and INA3221 both default to 0x40 — one must be jumpered to an alternate address during assembly.

---

*Last updated: February 2026*
*Cerberus doesn't ask for permission. It logs what it did and moves on.*