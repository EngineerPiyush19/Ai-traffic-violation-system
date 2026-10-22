# AI Traffic Violation System

One camera, three jobs: **AI traffic-signal control**, **license plate logging**, and **red-light violation detection**.

> **Disclaimer:** This is a prototype / academic project, not an official enforcement system. Automated detections can be wrong, and every violation should be reviewed by a human before any action is taken.

## Demo

<!-- Add a screenshot or GIF here, e.g. ![Demo](demo/demo.gif). Blur or fake any real plates. -->

## What It Does

1. **Traffic control:** YOLOv8 tracks vehicles in four approach zones (north, east, south, west). Vehicle counts and queue lengths are converted into a SUMO-RL style observation, and a trained **DQN agent** chooses the signal phase.
2. **Plate logging:** a plate detector and `fast-plate-ocr` read license plates. A plate is logged only after several confirming reads, together with the approach and signal phase.
3. **Red-light check:** you draw a stop line on the video. A vehicle moving left to right that crosses the line while its approach is RED (according to the DQN signal) is logged as a violation, with a snapshot image and its plate.

## Tech Stack

| Part | Tool |
|---|---|
| Language | Python |
| Vehicle detection and tracking | YOLOv8 (Ultralytics) + ByteTrack |
| Video processing | OpenCV |
| Signal control | DQN (Stable-Baselines3), SUMO-RL style observation |
| Plate OCR | fast-plate-ocr |
| Training environment | SUMO + sumo-rl (only needed to retrain the agent) |

## Project Structure

```
ai-traffic-violation-system/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── run_live_sumorl_violations.py   # main script: control + plates + violations
├── plates.py                       # standalone plate reader
├── plates.pt                       # trained plate detector
├── stop_line.json                  # example stop line (created by pressing L)
├── outputs/
│   └── 2way-single-intersection/
│       └── dqn_model.zip           # trained DQN agent
└── demo/                           # screenshots / GIF
```

## Installation

```bash
git clone https://github.com/<your-username>/ai-traffic-violation-system.git
cd ai-traffic-violation-system
python -m venv venv
venv\Scripts\activate         # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

You also need `plates.pt` (plate detector) and `outputs/2way-single-intersection/dqn_model.zip` (trained agent). YOLOv8 vehicle weights (`yolov8n.pt`) download automatically on first run.

## Usage

**Full system**

```bash
# Video file
py run_live_sumorl_violations.py --source traffic.mp4 --agent outputs/2way-single-intersection/dqn_model.zip

# Webcam, without plate reading
py run_live_sumorl_violations.py --source 0 --agent outputs/2way-single-intersection/dqn_model.zip --no-plates
```

**Controls in the video window**

| Key | Action |
|---|---|
| `L` | then click 2 points to set the stop line (saved to `stop_line.json`) |
| click | print pixel coordinates |
| `ESC` / `q` | quit |

**Useful options**

| Option | Purpose |
|---|---|
| `--process-width 640` | resize frames before detection (faster, lower resolution) |
| `--green-phases 2,3` | signal phases in which the monitored approach is green |
| `--debug-plates` | print every OCR read |
| `--no-plates` | skip plate reading |

**Standalone plate reader (optional)**

`plates.py` runs only the plate detector and OCR on a video, with no signal control or violation logic. Edit `SOURCE` at the top of the file, then run:

```bash
py plates.py
```

Crops are saved in `crops/`, and plates seen at least twice are printed at the end.

## Outputs

| File / folder | Contents |
|---|---|
| `plates_traffic_log.csv` | confirmed plates with approach and signal phase |
| `violations_log.csv` | red-light violations: time, track id, vehicle, phase, state, plate, image |
| `violations/` | one snapshot image per violation |
| `crops/` | plate crops |

## Configuration

Lane zones, signal timing (`DELTA_TIME`, `YELLOW_TIME`, `MIN_GREEN`) and the plate format (Indian pattern, e.g. `DL3CCR8958`) are set at the top of `run_live_sumorl_violations.py`. Lane zones are pixel boxes and must be adjusted to match your camera view.


## Limitations

- Lane zones and the stop line are set manually for one camera view.
- Plate reading is less accurate at night, in rain, and with small or angled plates.
- Violation logic currently covers left-to-right (eastbound) traffic only.
- Signal state comes from the DQN agent, not from a real traffic light.

## Ethical Considerations

- Automated enforcement can make mistakes; a person should review every violation.
- License plates are personal data. Do not publish real footage, plate crops or logs without anonymizing them.

## Future Work

- Violation checks for all approaches
- Additional violations (speeding, helmet, wrong lane)
- PDF ticket generation and a review dashboard
- Splitting the main script into separate modules

## License

MIT License. See `LICENSE`.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [SUMO-RL](https://github.com/LucasAlegre/sumo-rl)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
- [OpenCV](https://opencv.org/)
