
# AI Traffic System

An AI-based traffic monitoring system that detects vehicles and traffic violations from video, reads license plates, and automatically generates violation tickets.

> **Disclaimer:** This is a prototype / academic project. It is not an official enforcement system. Automated detections can be wrong and should be reviewed by a human before any ticket is issued.

## Demo

<!-- Add a GIF or screenshot here, e.g. ![Demo](demo/demo.gif) -->

Sample ticket (dummy data): see `demo/sample_ticket.pdf`

## Features

- Vehicle detection using YOLOv8
- Violation detection: <!-- e.g. red-light jumping, over-speeding, no helmet, wrong lane -->
- License plate detection and text recognition (EasyOCR)
- Automatic violation ticket generation
- Configurable rules (speed limits, fine amounts) via `config.yaml`

## How It Works

```
Video / Camera feed
        |
        v
Vehicle detection (YOLOv8)
        |
        v
Violation check (rules in config.yaml)
        |
        v
Plate detection + OCR (EasyOCR)
        |
        v
Violation ticket generated
```

## Tech Stack

| Part | Tool |
|---|---|
| Language | Python |
| Object detection | YOLOv8 (Ultralytics) |
| Video / image processing | OpenCV |
| Plate text recognition | EasyOCR |

## Installation

```bash
git clone https://github.com/<your-username>/ai-traffic-system.git
cd ai-traffic-system
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --source path/to/video.mp4
```

<!-- Replace with your actual entry point and arguments -->

Output tickets are saved in the `output/` folder.

## Project Structure

```
ai-traffic-system/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── main.py
├── src/
│   ├── detection/
│   ├── violations/
│   ├── plate_recognition/
│   └── ticketing/
├── models/
├── data/sample/
├── demo/
└── docs/
```

## Results

| Metric | Value |
|---|---|
| Vehicle detection mAP | <!-- fill in --> |
| Plate recognition accuracy | <!-- fill in --> |
| Processing speed (FPS) | <!-- fill in --> |

**Known limitations:** accuracy drops at night, in heavy rain, and with blurry or angled plates.

## Ethical Considerations

- Automated enforcement can make mistakes; tickets should be reviewed by a person before being issued.
- License plates are personal data. Do not publish real footage or tickets without anonymizing them.
- This project uses only test or publicly licensed data.

## Future Work

- Emergency vehicle priority
- Multi-camera / multi-junction support
- Web dashboard for reviewing tickets

## License

MIT License. See `LICENSE`.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)
