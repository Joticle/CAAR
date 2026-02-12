# CERBERUS — Phase 1 Build Manual

**Operation Ground Truth — First Boot to First Roll**

This manual covers the complete first-week build: Pi 5 setup, chassis assembly, power system wiring, motor driver hookup, sensor integration, and full system validation. Every step references exact GPIO pins, I2C addresses, and config values from `cerberus.yaml`.

> **Safety First:** You are working with lithium batteries (11.1V, 50C discharge), high-current motor drivers (43A rated), and live electronics. Disconnect battery power before making any wiring changes. Keep the LiPo safe bag within arm's reach whenever batteries are out of storage. Las Vegas heat amplifies every thermal risk.

---

## Prerequisites

### Hardware On Hand

| Item | Needed For |
|------|-----------|
| Raspberry Pi 5 (8GB) | Steps 1–3 |
| 128GB microSD (Samsung EVO Plus) | Step 1 |
| Argon ONE V3 Case | Step 2 |
| USB-C power supply (any 5V/3A+ temporary) | Step 1 |
| Windows PC with VS Code | All steps |
| Wi-Fi network (2.4GHz or 5GHz) | Step 1 |
| Soldering station (WEP 927-IV) | Steps 5, 7, 8 |
| Multimeter | Steps 5–9 |

### Hardware Arriving This Week

| Item | Needed For |
|------|-----------|
| 6WD Chassis Kit (Seeed Studio) | Step 4 |
| BTS7960 Motor Drivers x2 | Step 6 |
| Zeee 3S Batteries x2 | Step 5 |
| HTRC T150 Charger | Step 5 |
| XL4015 Buck Converter | Step 5 |
| INA3221 Power Monitor x2 | Step 7 |
| Pi Camera Module 3 | Step 8 |
| Pi 5 Camera Cable | Step 8 |
| NEO-6M GPS Module | Step 8 |
| HC-SR04 Ultrasonic Sensors | Step 8 |
| PCA9685 Servo Driver | Step 7 |
| Fuse Block (6-way) | Step 5 |
| Breadboards, jumper wires, standoffs | Steps 6–9 |

### Software On Hand

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11.9 | Matches Pi OS Bookworm target |
| VS Code | Latest | IDE + Remote-SSH |
| Git | Latest | Repo management |
| Raspberry Pi Imager | Latest | [Download](https://www.raspberrypi.com/software/) |

---

## Step 1: Flash and First Boot

**Time estimate:** 30 minutes  
**What you need:** Pi 5, microSD card, USB-C power supply, Windows PC

### 1.1 — Flash Raspberry Pi OS

1. Insert the microSD card into your Windows PC
2. Open **Raspberry Pi Imager**
3. Click **Choose Device** → select **Raspberry Pi 5**
4. Click **Choose OS** → select **Raspberry Pi OS Lite (64-bit)** under "Raspberry Pi OS (other)"
   - This is the Bookworm release, headless, no desktop — exactly what Cerberus needs
5. Click **Choose Storage** → select your microSD card
6. Click **Next** — when prompted to customize, click **Edit Settings**

### 1.2 — Pre-Configure (Critical)

In the **General** tab:
- **Hostname:** `cerberus`
- **Username:** `cerberus` (or your preferred username)
- **Password:** set a strong password — write it down
- **Wi-Fi SSID:** your home network name
- **Wi-Fi Password:** your network password
- **Wi-Fi Country:** US
- **Locale:** America/Los_Angeles timezone

In the **Services** tab:
- **Enable SSH:** Yes
- **Use password authentication:** Yes

Click **Save**, then **Yes** to apply settings, then **Yes** to write. Wait for the flash and verification to complete.

### 1.3 — First Boot

1. Remove microSD from PC
2. Insert microSD into the Pi 5's card slot (underside of the board)
3. Plug in USB-C power — any 5V/3A+ adapter works temporarily
4. Wait 60–90 seconds for first boot (the Pi expands the filesystem on first run)

### 1.4 — Find Cerberus on Your Network

Open PowerShell or VS Code terminal on your Windows PC:

```
ping cerberus.local
```

If that resolves, you're connected. If not:
- Log into your router's admin page
- Check the DHCP client list for a device named `cerberus`
- Note the IP address (e.g., `192.168.1.XXX`)

### 1.5 — SSH In

```
ssh cerberus@cerberus.local
```

Accept the fingerprint prompt, enter your password. You should see a Debian/Bookworm terminal prompt.

**Verify Python:**
```bash
python3 --version
```
Expected output: `Python 3.11.x`

**Verify architecture:**
```bash
uname -m
```
Expected output: `aarch64` (confirms 64-bit OS)

### 1.6 — System Update

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Wait 30 seconds, SSH back in.

---

## Step 2: Argon ONE V3 Case Assembly

**Time estimate:** 20 minutes  
**What you need:** Argon ONE V3 case, Pi 5, small Phillips screwdriver

### 2.1 — Unbox and Inspect

The Argon ONE V3 ships with:
- Top aluminum shell (heatsink)
- Bottom plastic base
- GPIO bridge board + ribbon cable
- Thermal pads (pre-applied or separate)
- Screws
- Power button and IR receiver (built into case)

### 2.2 — Install the Pi 5

1. **Power off the Pi** — `sudo shutdown -h now`, then unplug USB-C
2. Remove the microSD card (you'll reinsert it after the case is assembled)
3. If thermal pads aren't pre-applied, place them on the CPU and RAM chips
4. Seat the Pi 5 onto the standoffs in the bottom case half
5. Connect the GPIO bridge board ribbon cable to the Pi 5's 40-pin header
6. Secure with the provided screws
7. Snap the aluminum top onto the bottom — the thermal pads should make contact with the CPU/RAM through the top shell

### 2.3 — Reassemble and Boot

1. Reinsert the microSD card
2. Plug in USB-C power through the case's rear USB-C port
3. Press the **power button** on the case
4. SSH back in to verify everything still works

### 2.4 — Install Argon Fan Control (Optional)

```bash
curl https://download.argon40.com/argon1.sh | bash
```

This installs the fan speed controller daemon. Default thresholds:
- 55°C → 10% fan speed
- 60°C → 55% fan speed  
- 65°C → 100% fan speed

These are reasonable for Vegas. The case's thermal management will be critical when Cerberus is running inference outdoors.

### 2.5 — Verify Thermals

```bash
vcgencmd measure_temp
```

At idle indoors you should see 35–45°C. If you see 60°C+ at idle, recheck thermal pad contact.

---

## Step 3: Deploy the Codebase

**Time estimate:** 15 minutes  
**What you need:** Pi 5 running and SSH accessible

### 3.1 — Install System Dependencies

```bash
sudo apt install -y git python3-pip python3-venv i2c-tools
```

### 3.2 — Enable Hardware Interfaces

```bash
sudo raspi-config
```

Navigate to **Interface Options** and enable:
- **I2C** — for INA3221, PCA9685, BME680, SCD40, SHT31, QMC5883L
- **SPI** — for BME680 (optional secondary interface)
- **Serial Port** — for GPS (disable login shell, enable hardware)
- **Camera** — (legacy option, may not appear on Bookworm — libcamera is default)

Reboot:
```bash
sudo reboot
```

### 3.3 — Create the Cerberus Directory

```bash
sudo mkdir -p /opt/cerberus
sudo chown $USER:$USER /opt/cerberus
```

### 3.4 — Clone the Repository

```bash
git clone https://github.com/Joticle/CAAR.git /opt/cerberus
cd /opt/cerberus
```

### 3.5 — Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

**Important:** Every time you SSH in to work on Cerberus, activate the venv first:
```bash
cd /opt/cerberus && source venv/bin/activate
```

### 3.6 — Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### 3.7 — Install Pi-Specific Dependencies

These packages only work on the Pi (they'll fail on your Windows dev machine):

```bash
pip install gpiozero lgpio picamera2 tflite-runtime gpsd-py3
pip install adafruit-circuitpython-bme680 adafruit-circuitpython-scd4x
pip install smbus2
```

> **Note:** Some of these may fail if the corresponding hardware isn't connected yet. That's fine — the code handles missing hardware gracefully.

### 3.8 — Create Runtime Directories

```bash
mkdir -p data/audio/predator
mkdir -p data/weed_detections
mkdir -p data/surveillance
mkdir -p data/pest_detections
mkdir -p data/bird_photos
mkdir -p data/microclimate
```

### 3.9 — Run the Test Suite

```bash
pytest tests/ -v
```

**All 200+ tests should pass.** They are designed to run without hardware connected. This validates your entire codebase on the target platform.

If any test fails, note the exact error — it likely indicates a missing pip package or a Python version mismatch.

### 3.10 — Run the Standalone Smoke Test

```bash
python scripts/smoke_test.py --verbose
```

This exercises the full boot sequence: config loading → database creation → logger setup → health monitor → MQTT client → safety watchdog.

---

## Step 4: Chassis Assembly

**Time estimate:** 2–3 hours  
**What you need:** 6WD chassis kit, Phillips screwdriver, hex wrenches (included in kit)

### 4.1 — Unbox and Inventory

The Seeed Studio 6WD kit should include:
- 2x aluminum deck plates (upper and lower)
- 6x DC motors with gearboxes (1:34 ratio, 17000 RPM original / ~500 RPM after gearing)
- 6x wheels with rubber tires
- 6x suspension cantilever arms
- 6x shock absorbers (spring type)
- 6x motor mounting brackets
- 6x wheel couplings (shaft adapters)
- 4x copper standoff posts
- Assorted M3/M4 screws, nuts, and hex bolts
- Power switch

**Photograph everything before assembly.** If a part is missing or damaged, you'll need the photos for the seller.

### 4.2 — Build Sequence

Follow the assembly guide included with the kit (also available as the PDF in this repo). The build order is:

1. **Mount brackets** to lower deck plate — 6 brackets, screws from underneath
2. **Install cantilever joints** — pivot bearings for each suspension arm
3. **Attach cantilever arms** — 6 arms, one per motor position
4. **Mount couplings** — shaft adapters connect motor shafts to wheel hubs
5. **Install shock absorbers** — spring shocks between deck and cantilever arms
6. **Mount wheels** — press/screw onto couplings
7. **Install upper deck** — copper standoff posts between lower and upper plates
8. **Mount power switch** — through the designated cutout

### 4.3 — Motor Wiring Convention

**Establish this now and label every wire.** Cerberus uses skid-steer (tank drive):

| Position | Side | Motor Wires | BTS7960 |
|----------|------|-------------|---------|
| Front Left | LEFT | Red (+) / Black (-) | Driver 1 |
| Middle Left | LEFT | Red (+) / Black (-) | Driver 1 |
| Rear Left | LEFT | Red (+) / Black (-) | Driver 1 |
| Front Right | RIGHT | Red (+) / Black (-) | Driver 2 |
| Middle Right | RIGHT | Red (+) / Black (-) | Driver 2 |
| Rear Right | RIGHT | Red (+) / Black (-) | Driver 2 |

**All 3 left motors wire in parallel to BTS7960 Driver 1.**  
**All 3 right motors wire in parallel to BTS7960 Driver 2.**

Use masking tape or heat shrink labels. Write "L" or "R" on each motor wire pair now. You will thank yourself later.

### 4.4 — Motor Direction Test (Manual)

Before wiring to drivers, verify motor spin direction:
1. Briefly touch a single motor's wires to a AA battery or bench supply
2. Note which wire polarity makes the wheel spin **forward** (away from you when looking from behind the rover)
3. If a motor spins the wrong way, swap its red/black wires
4. **All left motors must spin the same direction for forward**
5. **All right motors must spin the same direction for forward**
6. Left and right sides will spin **opposite** to each other for forward motion — that's correct for skid-steer

---

## Step 5: Power System

**Time estimate:** 1–2 hours  
**What you need:** Batteries, charger, buck converter, fuse block, multimeter, XT60 connectors, wire

> **⚠️ DANGER — READ BEFORE PROCEEDING**
> - Never short-circuit a LiPo/Li-ion battery — instant fire risk
> - Never charge unattended — use the LiPo safe bag
> - Always disconnect the battery before wiring changes
> - Verify polarity with a multimeter BEFORE connecting anything
> - In Las Vegas summer, never store batteries in a vehicle or garage above 100°F

### 5.1 — Charge Your Batteries

1. Place battery inside the **LiPo safe bag** during charging
2. Connect the Zeee 3S battery to the HTRC T150 charger
3. Connect the **balance lead** (small white connector) to the charger's balance port
4. Set charger to: **Li-ion / 3S / 2.0A charge rate**
5. Start charging — the charger will balance all 3 cells and stop automatically
6. Fully charged = **12.6V** (4.2V per cell)
7. Charge both batteries

### 5.2 — Fuse Block Wiring

The RED WOLF 6-way fuse block provides organized, protected power distribution.

| Fuse Slot | Circuit | Fuse Rating | Wire Gauge |
|-----------|---------|-------------|------------|
| 1 | Battery Main (input) | 15A | 12 AWG |
| 2 | Pi 5 (via buck converter) | 5A | 18 AWG |
| 3 | Left Motor Driver (BTS7960) | 10A | 14 AWG |
| 4 | Right Motor Driver (BTS7960) | 10A | 14 AWG |
| 5 | Payload Head Power | 5A | 18 AWG |
| 6 | Spare | — | — |

**Wiring the fuse block:**
1. Battery XT60 → main power switch → Fuse Slot 1 (input)
2. Fuse block common ground bus → all ground connections
3. Each output slot feeds its designated circuit

### 5.3 — Buck Converter Setup (XL4015)

The XL4015 steps down battery voltage (9.6–12.6V) to a stable 5V for the Pi 5.

**CRITICAL — Set output voltage BEFORE connecting the Pi:**

1. Connect battery to the buck converter input terminals (VIN+ / VIN-)
2. **Do NOT connect anything to the output yet**
3. Use your multimeter on the output terminals (VOUT+ / VOUT-)
4. Adjust the onboard potentiometer (small brass screw) until output reads **5.10V**
   - Slightly above 5.0V compensates for wire voltage drop
   - **Never exceed 5.25V** — this is the Pi 5's absolute max
5. Verify the output holds steady for 30 seconds
6. Disconnect battery
7. Label the converter: "5V PI POWER — DO NOT ADJUST"

### 5.4 — Power Distribution Diagram

```
                    ┌──────────┐
  Battery XT60 ─────┤  SWITCH  ├─────┐
                    └──────────┘     │
                                     ▼
                              ┌──────────┐
                              │  FUSE    │
                              │  BLOCK   │
                              │          │
                    Slot 1 ───┤ 15A Main │
                    Slot 2 ───┤ 5A Pi    │──→ XL4015 → 5V → Pi 5 USB-C
                    Slot 3 ───┤ 10A Left │──→ BTS7960 #1 (L motors)
                    Slot 4 ───┤ 10A Right│──→ BTS7960 #2 (R motors)
                    Slot 5 ───┤ 5A Aux   │──→ Payload head sensors
                    Slot 6 ───┤ Spare    │
                              │          │
                    GND Bus ──┤ Common ──│──→ All grounds tied together
                              └──────────┘
```

### 5.5 — Verify Power with Multimeter

Before connecting any electronics:

| Test Point | Expected Reading | Action if Wrong |
|-----------|-----------------|-----------------|
| Battery terminals | 10.5–12.6V DC | Charge battery |
| Fuse block output (each slot) | Same as battery | Check fuse / wiring |
| Buck converter output | 5.10V ± 0.05V | Adjust potentiometer |
| Ground continuity | 0Ω between all grounds | Fix ground bus |

---

## Step 6: Motor Driver Wiring

**Time estimate:** 1–2 hours  
**What you need:** 2x BTS7960 drivers, jumper wires, multimeter

> **Disconnect battery before wiring anything in this step.**

### 6.1 — BTS7960 Overview

Each BTS7960 board has two sets of connections:

**Power side (screw terminals):**
- **B+** — Battery positive (from fuse block)
- **B-** — Battery negative (ground bus)
- **M+** — Motor positive output
- **M-** — Motor negative output

**Logic side (pin header):**
- **VCC** — 3.3V or 5V logic power (from Pi)
- **GND** — Logic ground (connect to Pi GND)
- **R_EN** — Right enable (connect to forward pin)
- **L_EN** — Left enable (connect to reverse pin)
- **RPWM** — Right PWM input (forward speed)
- **LPWM** — Left PWM input (reverse speed)
- **R_IS** / **L_IS** — Current sense outputs (optional)

### 6.2 — Driver 1: Left Side Motors

**Power connections (screw terminals):**
- **B+** → Fuse block Slot 3 output (10A)
- **B-** → Ground bus
- **M+** → All 3 left motors positive (parallel)
- **M-** → All 3 left motors negative (parallel)

**Logic connections (to Pi 5 GPIO):**

| BTS7960 Pin | Pi 5 GPIO | Config Reference |
|------------|-----------|-----------------|
| RPWM | GPIO 12 (PWM0) | `motors.left.pwm_pin: 12` |
| R_EN | GPIO 5 | `motors.left.forward_pin: 5` |
| L_EN | GPIO 6 | `motors.left.reverse_pin: 6` |
| LPWM | GPIO 12 (PWM0) | Same PWM pin — tied together |
| VCC | Pi 3.3V | Logic level power |
| GND | Pi GND | Logic ground — **must share ground with Pi** |

> **Note on PWM wiring:** The `motor_driver.py` module uses the enable pins for direction and a single PWM for speed. RPWM and LPWM can be tied to the same PWM pin — the enable pins determine which H-bridge half is active.

### 6.3 — Driver 2: Right Side Motors

**Power connections (screw terminals):**
- **B+** → Fuse block Slot 4 output (10A)
- **B-** → Ground bus
- **M+** → All 3 right motors positive (parallel)
- **M-** → All 3 right motors negative (parallel)

**Logic connections (to Pi 5 GPIO):**

| BTS7960 Pin | Pi 5 GPIO | Config Reference |
|------------|-----------|-----------------|
| RPWM | GPIO 13 (PWM1) | `motors.right.pwm_pin: 13` |
| R_EN | GPIO 16 | `motors.right.forward_pin: 16` |
| L_EN | GPIO 26 | `motors.right.reverse_pin: 26` |
| LPWM | GPIO 13 (PWM1) | Same PWM pin — tied together |
| VCC | Pi 3.3V | Logic level power |
| GND | Pi GND | Logic ground |

### 6.4 — Emergency Stop

| Function | Pi 5 GPIO | Config Reference |
|----------|-----------|-----------------|
| E-Stop Input | GPIO 25 | `motors.emergency_stop_pin: 25` |

Wire a normally-closed momentary switch between GPIO 25 and GND. The software reads this as **active low** — pressing the button pulls the pin low and triggers an immediate all-motor stop. For initial bench testing, leave this unconnected (the pin has an internal pull-up).

### 6.5 — Pre-Power Checklist

Before connecting battery power to the motor drivers:

- [ ] All motor wires are secure in screw terminals (no loose strands)
- [ ] Left motors go to Driver 1, right motors to Driver 2
- [ ] B+ goes to fused output, B- goes to ground bus
- [ ] Logic wires go to correct GPIO pins (double-check against table above)
- [ ] Pi and BTS7960 share a common ground
- [ ] No bare wire ends touching each other or the chassis
- [ ] Multimeter confirms no short between B+ and B- on each driver

---

## Step 7: I2C Device Wiring

**Time estimate:** 1 hour  
**What you need:** INA3221, PCA9685, breadboard, jumper wires, soldering station (for header pins)

### 7.1 — Solder Header Pins

Both the INA3221 and PCA9685 ship with **unsoldered header pins**. Solder them now:

1. Insert header pins into a breadboard to hold them straight
2. Place the board on top of the pins
3. Solder each pin — quick touch, don't overheat the board
4. Inspect for cold joints (dull/grainy solder = redo)

### 7.2 — I2C Address Conflict Resolution

**This is critical.** Both devices default to address `0x40`:

| Device | Default Address | Config Reference |
|--------|----------------|-----------------|
| INA3221 | 0x40 | `power.i2c_address: 0x40` |
| PCA9685 | 0x40 | `servos.i2c_address: 0x40` |

**Fix:** Jumper the PCA9685's A0 address pad to change it to **0x41**.

On the PCA9685 board, find the address solder jumpers (labeled A0–A5). Bridge **A0** with a small solder blob. This changes the address to `0x41`.

**Update `cerberus.yaml`** after jumpering:
```yaml
servos:
  i2c_address: 0x41    # Changed from 0x40 to avoid INA3221 conflict
```

### 7.3 — I2C Wiring (Shared Bus)

All I2C devices share the same two data lines. The Pi 5 I2C bus 1 is on:

| Signal | Pi 5 GPIO | Physical Pin |
|--------|-----------|-------------|
| SDA | GPIO 2 | Pin 3 |
| SCL | GPIO 3 | Pin 5 |
| 3.3V | — | Pin 1 |
| GND | — | Pin 6, 9, 14, 20, 25, 30, 34, 39 |

**Wiring each I2C device (same 4 wires for all):**

| Device Wire | Connects To |
|-------------|-------------|
| VCC / VIN | Pi 3.3V (Pin 1) or 5V (Pin 2) — check device datasheet |
| GND | Pi GND (any ground pin) |
| SDA | Pi GPIO 2 (Pin 3) |
| SCL | Pi GPIO 3 (Pin 5) |

### 7.4 — I2C Bus Map (All Devices)

| Address | Device | Voltage | Status |
|---------|--------|---------|--------|
| 0x0D | QMC5883L Compass | 3.3V | Connect in Step 8 |
| 0x40 | INA3221 Power Monitor | 3.3V | Connect now |
| 0x41 | PCA9685 Servo Driver | 5V (logic 3.3V) | Connect now (after A0 jumper) |
| 0x44 | SHT31-D Temp/Humidity | 3.3V | Connect in Step 8 |
| 0x62 | SCD40 CO2 Sensor | 3.3V | Connect in Step 8 |
| 0x77 | BME680 Environmental | 3.3V | Connect in Step 8 |

### 7.5 — Verify I2C Devices

After wiring the INA3221 and PCA9685:

```bash
sudo i2cdetect -y 1
```

Expected output shows devices at their addresses:
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
40: 40 41 -- -- -- -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- --
```

`40` = INA3221, `41` = PCA9685. If either is missing, check wiring and solder joints.

---

## Step 8: Sensor Integration

**Time estimate:** 1–2 hours  
**What you need:** Camera, GPS, HC-SR04, compass, environmental sensors, jumper wires

### 8.1 — Pi Camera Module 3

1. **Power off the Pi** — `sudo shutdown -h now`, unplug power
2. Locate the **CAM/DSI** connector on the Pi 5 (between the HDMI ports and the 3.5mm jack area — it's a smaller 22-pin connector on Pi 5)
3. Use the **22-pin to 15-pin FPC cable** (the Pi Camera 3 standard cable is 15-pin on the camera end and needs the 22-pin adapter for Pi 5)
4. Lift the connector latch gently, insert the cable with contacts facing the correct direction, close the latch
5. Mount the camera — use tape or a temporary stand for now
6. Power on and test:

```bash
rpicam-still -o test.jpg
```

If you see a JPEG file, the camera works. Transfer it to your PC to verify:
```bash
# From your Windows PC:
scp cerberus@cerberus.local:/opt/cerberus/test.jpg .
```

### 8.2 — GPS Module (NEO-6M)

**Wiring:**

| NEO-6M Pin | Pi 5 Connection | Notes |
|-----------|----------------|-------|
| VCC | Pi 5V (Pin 2) | Module has onboard 3.3V regulator |
| GND | Pi GND | |
| TX | Pi GPIO 15 (RXD, Pin 10) | GPS TX → Pi RX (crossed) |
| RX | Pi GPIO 14 (TXD, Pin 8) | GPS RX → Pi TX (crossed) |

**Software setup:**

```bash
sudo apt install -y gpsd gpsd-clients

# Configure gpsd to use the Pi's serial port
sudo nano /etc/default/gpsd
```

Set these values:
```
START_DAEMON="true"
GPSD_OPTIONS="-n"
DEVICES="/dev/ttyAMA0"
USBAUTO="false"
```

Restart and test:
```bash
sudo systemctl restart gpsd
gpsmon
```

**First fix takes 1–5 minutes outdoors.** The NEO-6M needs clear sky visibility. If testing indoors, GPS won't lock — that's expected. The config supports simulation mode for indoor testing:

```yaml
gps:
  simulation:
    enabled: true     # Set to true for indoor testing
    lat: 36.1699
    lon: -115.1398
```

### 8.3 — Ultrasonic Sensors (HC-SR04)

Each HC-SR04 needs **two GPIO pins** (trigger and echo) plus power:

| Wire | Connection |
|------|-----------|
| VCC | Pi 5V (Pin 2) |
| GND | Pi GND |
| TRIG | Dedicated GPIO (output) |
| ECHO | Dedicated GPIO (input) — **needs voltage divider** |

> **⚠️ VOLTAGE WARNING:** The HC-SR04 ECHO pin outputs 5V, but Pi 5 GPIO is 3.3V only. You **must** use a voltage divider (1kΩ + 2kΩ resistors) on each ECHO line, or use the 3.3V-compatible HC-SR04P variant. Applying 5V to a Pi GPIO pin will damage the RP1 chip permanently.

**Suggested GPIO assignments for 3-sensor front array:**

| Sensor | TRIG GPIO | ECHO GPIO | Position |
|--------|----------|----------|----------|
| Left | GPIO 17 | GPIO 27 | Front-left 45° |
| Center | GPIO 22 | GPIO 10 | Front-center |
| Right | GPIO 9 | GPIO 11 | Front-right 45° |

For bench testing, wire one sensor first and verify with:
```bash
python3 -c "
from gpiozero import DistanceSensor
sensor = DistanceSensor(echo=27, trigger=17)
print(f'Distance: {sensor.distance * 100:.1f} cm')
"
```

### 8.4 — Compass Module (QMC5883L)

Standard I2C wiring to the shared bus:

| QMC5883L Pin | Pi 5 Connection |
|-------------|----------------|
| VCC | Pi 3.3V (Pin 1) |
| GND | Pi GND |
| SDA | Pi GPIO 2 (Pin 3) |
| SCL | Pi GPIO 3 (Pin 5) |

Verify it appears on the bus:
```bash
sudo i2cdetect -y 1
```

Look for `0d` in the output — that's the QMC5883L at address 0x0D.

### 8.5 — Environmental Sensors

Wire each to the shared I2C bus (same SDA/SCL/3.3V/GND as above). Connect them one at a time and verify each with `i2cdetect`:

| Sensor | Expected Address | Verify With |
|--------|-----------------|-------------|
| BME680 | 0x77 | `sudo i2cdetect -y 1` → look for `77` |
| SCD40 | 0x62 | `sudo i2cdetect -y 1` → look for `62` |
| SHT31-D | 0x44 | `sudo i2cdetect -y 1` → look for `44` |

**Full bus scan after all sensors connected:**
```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- 0d -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
40: 40 41 -- -- 44 -- -- -- -- -- -- -- -- -- -- --
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
60: -- -- 62 -- -- -- -- -- -- -- -- -- -- -- -- --
70: -- -- -- -- -- -- -- 77
```

Seven devices on the bus: INA3221, PCA9685, SHT31, SCD40, QMC5883L, BME680. All unique addresses, no conflicts.

---

## Step 9: Full System Validation

**Time estimate:** 1 hour  
**What you need:** Everything wired, battery charged, clear outdoor area for GPS

### 9.1 — VS Code Remote-SSH Setup

On your Windows PC, set up VS Code for remote development:

1. Install the **Remote-SSH** extension in VS Code
2. Press `Ctrl+Shift+P` → **Remote-SSH: Connect to Host**
3. Enter: `cerberus@cerberus.local`
4. VS Code opens a remote window connected to the Pi
5. Open folder: `/opt/cerberus`

You can now edit code on the Pi directly from VS Code on your Windows PC.

### 9.2 — I2C Bus Audit

```bash
cd /opt/cerberus && source venv/bin/activate
sudo i2cdetect -y 1
```

Compare the output against the I2C Bus Map in Step 7.4. Every expected device should appear.

### 9.3 — Config Validator

```bash
python3 -c "
from cerberus.core.config import Config
from cerberus.core.config_validator import ConfigValidator
config = Config('config/cerberus.yaml')
validator = ConfigValidator(config)
results = validator.validate_all()
for r in results:
    print(f'{r.level}: {r.message}')
"
```

This checks all GPIO pin assignments for conflicts, validates I2C addresses, and verifies threshold ordering (warn < critical < shutdown). Fix any errors it reports before proceeding.

### 9.4 — Motor Test Script

```bash
python3 scripts/test_motors.py
```

This script should run each side briefly at low speed. **Keep the rover elevated** (wheels off the ground) for this test. Verify:
- [ ] Left side motors all spin the same direction on "forward"
- [ ] Right side motors all spin the same direction on "forward"
- [ ] Left and right spin opposite directions for forward motion (skid-steer)
- [ ] Motors stop cleanly on command
- [ ] No grinding, clicking, or stalling

If a side spins the wrong direction, swap the M+ and M- wires on that BTS7960's screw terminals.

### 9.5 — Camera Validation

```bash
python3 -c "
from cerberus.perception.camera import Camera
cam = Camera()
cam.start()
frame = cam.capture_frame()
print(f'Frame shape: {frame.shape}')
cam.stop()
"
```

Expected output: `Frame shape: (480, 640, 3)` (inference resolution from config).

### 9.6 — Power Monitor Check

```bash
python3 -c "
import smbus2
bus = smbus2.SMBus(1)
# INA3221 manufacturer ID register
mid = bus.read_word_data(0x40, 0xFE)
print(f'INA3221 Manufacturer ID: {hex(mid)}')
bus.close()
"
```

If this returns a hex value (typically `0x5449`), the INA3221 is communicating correctly.

### 9.7 — GPS Lock Test (Outdoors)

Take the rover (or just the Pi + GPS module on a long USB-C cable) outside:

```bash
gpsmon
```

Wait for a fix. You should see latitude, longitude, altitude, and satellite count populate. The config requires a minimum of 4 satellites (`gps.min_satellites: 4`) and HDOP below 5.0 (`gps.max_hdop: 5.0`).

### 9.8 — Full Smoke Test

```bash
python scripts/smoke_test.py --verbose
```

This is the final validation. It exercises every subsystem in the correct boot order. All checks should pass.

### 9.9 — Install as System Service (Optional)

When you're ready for Cerberus to auto-start on boot:

```bash
sudo cp /opt/cerberus/cerberus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cerberus
sudo systemctl start cerberus
```

Check status:
```bash
sudo systemctl status cerberus
journalctl -u cerberus -f
```

> **Recommendation:** Don't enable auto-start until all hardware is validated. During development, run Cerberus manually so you can see output directly.

---

## GPIO Pin Map (Complete)

Quick reference for all GPIO assignments from `cerberus.yaml`:

| GPIO | Function | Direction | Config Key |
|------|----------|-----------|-----------|
| 2 | I2C SDA | Bidirectional | Hardware I2C1 |
| 3 | I2C SCL | Output | Hardware I2C1 |
| 5 | Left Motor Forward | Output | `motors.left.forward_pin` |
| 6 | Left Motor Reverse | Output | `motors.left.reverse_pin` |
| 9 | Ultrasonic Right TRIG | Output | obstacle config |
| 10 | Ultrasonic Center ECHO | Input | obstacle config |
| 11 | Ultrasonic Right ECHO | Input | obstacle config |
| 12 | Left Motor PWM | PWM0 | `motors.left.pwm_pin` |
| 13 | Right Motor PWM | PWM1 | `motors.right.pwm_pin` |
| 14 | GPS TXD (Pi → GPS) | Output | Serial UART |
| 15 | GPS RXD (GPS → Pi) | Input | Serial UART |
| 16 | Right Motor Forward | Output | `motors.right.forward_pin` |
| 17 | Ultrasonic Left TRIG | Output | obstacle config |
| 18 | NeoPixel Status LEDs | PWM | `leds.status.pin` |
| 22 | Ultrasonic Center TRIG | Output | obstacle config |
| 23 | IR LED Control | Output | `leds.ir.pin` |
| 24 | Pest Deterrent Eye LEDs | Output | `leds.pest_eyes.pin` |
| 25 | Emergency Stop | Input (pull-up) | `motors.emergency_stop_pin` |
| 26 | Right Motor Reverse | Output | `motors.right.reverse_pin` |
| 27 | Ultrasonic Left ECHO | Input | obstacle config |

**Unused GPIO available:** 0, 1, 4, 7, 8, 19, 20, 21

---

## Troubleshooting

### Pi Won't Boot
- Reflash the microSD — bad writes happen
- Try a different USB-C power supply (needs 5V/3A minimum)
- Remove all peripherals and try bare-board boot

### SSH Connection Refused
- Verify SSH was enabled in Pi Imager settings
- Check that Pi and PC are on the same network
- Try IP address directly instead of `cerberus.local`

### I2C Device Not Detected
- Check wiring: SDA to SDA, SCL to SCL (not crossed like serial)
- Verify 3.3V power reaching the device with multimeter
- Check solder joints on header pins
- Some devices need a pull-up resistor on SDA/SCL (the Pi has built-in 1.8kΩ pull-ups — usually sufficient)

### Motors Don't Spin
- Verify battery is charged (measure with multimeter: should be 10.5V+)
- Check fuse — is it blown?
- Verify BTS7960 power LED is on
- Check that Pi GND is connected to BTS7960 GND
- Run `test_motors.py` with `--verbose` flag for detailed output

### Camera Not Found
- Ensure cable is fully seated and latch is closed
- Pi 5 uses the smaller 22-pin connector — verify correct cable
- Run `rpicam-hello` to test basic camera connectivity
- Check `dmesg | grep -i camera` for kernel-level errors

### GPS No Fix
- Must be outdoors with clear sky visibility
- First fix after cold start takes 1–5 minutes
- Verify wiring: GPS TX → Pi RX (Pin 10), GPS RX → Pi TX (Pin 8)
- Check that gpsd is running: `sudo systemctl status gpsd`
- The ceramic antenna must face upward

---

## End of Phase 1

When all steps are complete, you should have:

- [x] Pi 5 running headless Raspberry Pi OS Lite (Bookworm 64-bit)
- [x] Cerberus codebase deployed and all tests passing
- [x] Argon ONE V3 case assembled with active cooling
- [x] 6WD chassis fully assembled with motors and wheels
- [x] Power system wired: battery → switch → fuse block → buck converter → Pi
- [x] Motor drivers wired and directionally verified
- [x] All I2C devices on the bus and responding
- [x] Camera capturing frames
- [x] GPS acquiring fix outdoors
- [x] Ultrasonic sensors reading distance
- [x] VS Code Remote-SSH connected for development

**Cerberus is alive. It can see, sense its position, detect obstacles, monitor its own power, and move.**

Next phase: AI model deployment, autonomous patrol routes, and field testing.

---

*Last updated: February 2026*
*Autonomous intelligence that lives where it works.*
