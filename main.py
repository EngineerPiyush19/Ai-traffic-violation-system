# run_live_sumorl_violations.py
"""
One camera, three jobs:
  1. Traffic control:  YOLO vehicle tracking -> SUMO-RL style observation -> DQN agent -> signal phase
  2. Plate logging:    plate detector + fast_plate_ocr -> confirmed plates logged with approach + signal phase
  3. Red-light check:  a stop line you draw on the video; a vehicle moving LEFT -> RIGHT that crosses it
                       while this approach is RED (according to the DQN signal) is logged as a violation.

Usage:
  py run_live_sumorl_violations.py --source traffic.mp4 --agent outputs/2way-single-intersection/dqn_model.zip
  py run_live_sumorl_violations.py --source 0 --agent outputs/2way-single-intersection/dqn_model.zip --no-plates

Keys / mouse in the video window:
  L        then click 2 points to set the STOP LINE (saved to stop_line.json, reused next run)
  click    print pixel coordinates in cmd
  ESC / q  quit

Outputs (in the folder you run from):
  plates_traffic_log.csv   confirmed plates
  violations_log.csv       red-light violations (time, track id, vehicle, phase, state, plate, image)
  violations\\              a snapshot image for each violation
  crops\\                   plate crops
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter, deque

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# TRAFFIC CONFIG
# =========================

# Lane order MUST match `lanes:` printed by your training script (same order the agent was trained with).
SUMO_LANES = [
    "n_t_0", "n_t_1",
    "e_t_0", "e_t_1",
    "s_t_0", "s_t_1",
    "w_t_0", "w_t_1",
]
LANE_TO_APPROACH = {"n": "north", "e": "east", "s": "south", "w": "west"}

N_GREEN_PHASES = 4
N_LANES = len(SUMO_LANES)
OBS_SHAPE = N_GREEN_PHASES + 1 + 2 * N_LANES   # phase one-hot + min_green flag + density + queue = 21

# sumo-rl defaults (change if you trained with different values)
DELTA_TIME = 5
YELLOW_TIME = 2
MIN_GREEN = 5

# Lane capacity used to normalise density/queue (use `lane lengths:` from training output)
LANE_LENGTH_M = 100.0
VEHICLE_SLOT_M = 7.5
LANE_CAPACITY = max(1.0, LANE_LENGTH_M / VEHICLE_SLOT_M)

# Camera zones (x1, y1, x2, y2) in pixels of the frame the script sees. Adjust after seeing your view.
LANE_ZONES = {
    "north": (250, 50, 450, 250),
    "east":  (450, 250, 650, 450),
    "south": (250, 450, 450, 650),
    "west":  (50, 250, 250, 450),
}

PHASE_NAMES = {0: "Phase 0 (NS)", 1: "Phase 1 (NS)", 2: "Phase 2 (EW)", 3: "Phase 3 (EW)"}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
HIST_LEN = 10      # frames used to decide a vehicle is stopped
STOP_PX = 6        # moved less than this many pixels => queued

# =========================
# RED-LIGHT CONFIG
# =========================

# Traffic here moves left -> right (east-bound), so this approach is green in the EW phases (2 and 3).
# Change with --green-phases if your phases are numbered differently.
DEFAULT_GREEN_PHASES = "2,3"
RED_GRACE_SEC = 1.0          # must have been red for this long before a crossing counts (avoids edge cases)
MIN_TRACK_FRAMES = 3         # vehicle must be tracked this many frames before it can be flagged
PLATE_WAIT_SEC = 3.0         # wait this long after a violation for the plate reader before writing the row
STOP_LINE_FILE = "stop_line.json"
VIOLATION_LOG = "violations_log.csv"

# =========================
# PLATE CONFIG
# =========================

MIN_PLATE_WIDTH = 60        # ignore plates narrower than this (pixels)
CONFIRM_READS = 2           # reads needed before a plate is logged
COOLDOWN_SEC = 30           # don't log the same plate again within this time
PLATE_EVERY = 2             # run plate detection on every Nth frame
PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")   # Indian format, e.g. DL3CCR8958
PLATE_LOG = "plates_traffic_log.csv"


def zone_of(cx, cy):
    for name, (x1, y1, x2, y2) in LANE_ZONES.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return name
    return None


# =========================
# VEHICLE COUNTING
# =========================

class VehicleCounter:
    def __init__(self, model):
        self.model = model
        self.history = {}

    def analyse(self, frame):
        results = self.model.track(frame, persist=True, verbose=False,
                                   tracker="bytetrack.yaml")[0]
        counts = {d: 0 for d in LANE_ZONES}
        queued = {d: 0 for d in LANE_ZONES}
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            self.history.clear()
            return counts, queued, results

        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        seen = set()
        for box, tid in zip(boxes, ids):
            if self.model.names[int(box.cls[0])] not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            direction = zone_of(cx, cy)
            if direction is None:
                continue
            counts[direction] += 1
            if tid is not None:
                seen.add(tid)
                hist = self.history.setdefault(tid, deque(maxlen=HIST_LEN))
                hist.append((cx, cy))
                if len(hist) == HIST_LEN:
                    moved = np.hypot(hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1])
                    if moved < STOP_PX:
                        queued[direction] += 1
        for tid in list(self.history):
            if tid not in seen:
                del self.history[tid]
        return counts, queued, results


# =========================
# PLATE READER
# =========================

class PlateReader:
    def __init__(self, model_path, debug=False, min_width=MIN_PLATE_WIDTH, conf=0.4):
        from fast_plate_ocr import LicensePlateRecognizer   # imported here so --no-plates works without it
        self.debug = debug
        self.min_width = min_width
        self.conf = conf
        self.status = "plates: waiting for first check"
        self.model = YOLO(model_path)
        self.ocr = LicensePlateRecognizer("cct-xs-v1-global-model")
        self.votes = Counter()
        self.last_logged = {}
        self.last_boxes = []
        self.fresh = []          # valid plate reads from the latest processed frame: (cx, cy, text)
        self.n = 0

        os.makedirs("crops", exist_ok=True)
        is_new = not os.path.exists(PLATE_LOG)
        self.log_file = open(PLATE_LOG, "a", newline="")
        self.writer = csv.writer(self.log_file)
        if is_new:
            self.writer.writerow(["time", "plate", "reads", "approach", "signal"])

    def process(self, frame, signal_text):
        """Call with the raw (un-drawn) frame."""
        self.n += 1
        self.fresh = []
        if self.n % PLATE_EVERY != 0:
            return
        self.last_boxes = []
        r = self.model(frame, imgsz=640, conf=self.conf, verbose=False)[0]
        n_det = len(r.boxes)
        n_small = 0

        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            if (x2 - x1) < self.min_width:
                n_small += 1
                self.last_boxes.append((x1, y1, x2, y2, f"small {x2 - x1}px", (0, 165, 255)))   # orange
                continue
            plate = frame[y1:y2, x1:x2]
            if plate.size == 0:
                continue

            res = self.ocr.run(plate)
            pred = res[0] if isinstance(res, (list, tuple)) and res else res
            text = getattr(pred, "plate", str(pred))
            text = "".join(ch for ch in text.upper() if ch.isalnum())
            if self.debug:
                print(f"OCR read '{text}' (box {x2 - x1}px wide) valid={bool(PATTERN.match(text))}")

            color = (0, 0, 255)                 # red = not a valid plate format
            if PATTERN.match(text):
                color = (0, 255, 0)             # green = valid format
                self.fresh.append(((x1 + x2) // 2, (y1 + y2) // 2, text))
                self.votes[text] += 1
                now = time.time()
                if self.votes[text] >= CONFIRM_READS and now - self.last_logged.get(text, 0) > COOLDOWN_SEC:
                    self.last_logged[text] = now
                    approach = zone_of((x1 + x2) // 2, (y1 + y2) // 2) or "unknown"
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.writer.writerow([stamp, text, self.votes[text], approach, signal_text])
                    self.log_file.flush()
                    cv2.imwrite(f"crops/{text}_{int(now)}.jpg", plate)
                    print(f"PLATE {stamp} {text} approach={approach} signal={signal_text}")
            self.last_boxes.append((x1, y1, x2, y2, text, color))
        self.status = f"plates: {n_det} detected, {n_small} too small (<{self.min_width}px)"

    def draw(self, frame):
        cv2.putText(frame, self.status, (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for (x1, y1, x2, y2, text, color) in self.last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, text, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    def close(self):
        self.log_file.close()


# =========================
# RED-LIGHT VIOLATION DETECTOR
# =========================

def approach_state(sig, green_phases):
    """State of the light for THIS approach (left -> right traffic): GREEN, YELLOW or RED.

    sig.phase is the phase currently shown; during a yellow, sig.phase is still the OLD phase.
    """
    if sig.yellow:
        if sig.phase in green_phases and sig.next_phase in green_phases:
            return "GREEN"
        if sig.phase in green_phases:
            return "YELLOW"          # this approach is being switched away from green
        return "RED"                 # this approach is still red while the other one clears
    return "GREEN" if sig.phase in green_phases else "RED"


class ViolationDetector:
    def __init__(self, green_phases):
        self.green = set(green_phases)
        self.line = None                 # ((x, y), (x, y))
        self.line_is_default = False
        self.prev_side = {}              # track id -> side of line on previous frame
        self.seen = Counter()            # track id -> frames tracked
        self.plate_votes = {}            # track id -> Counter of plate texts
        self.flagged = set()
        self.pending = {}                # track id -> info, waiting for the plate reader
        self.red_since = None
        self.count = 0
        self.flash_until = 0.0

        os.makedirs("violations", exist_ok=True)
        is_new = not os.path.exists(VIOLATION_LOG)
        self.file = open(VIOLATION_LOG, "a", newline="")
        self.writer = csv.writer(self.file)
        if is_new:
            self.writer.writerow(["time", "track_id", "vehicle", "signal_phase", "signal_state",
                                  "plate", "image"])

    # ---- stop line ----
    def load_or_default(self, frame_size):
        w, h = frame_size
        if os.path.exists(STOP_LINE_FILE):
            try:
                with open(STOP_LINE_FILE) as f:
                    data = json.load(f)
                if list(data["size"]) == [w, h]:
                    self.line = (tuple(data["line"][0]), tuple(data["line"][1]))
                    self.line_is_default = False
                    print(f"Loaded stop line from {STOP_LINE_FILE}: {self.line}")
                    return
                print("Saved stop line was for a different frame size, ignoring it.")
            except Exception as e:
                print("Could not read stop line file:", e)
        x = int(w * 0.6)
        self.line = ((x, 0), (x, h))
        self.line_is_default = True
        print("Using a default vertical stop line. Press L and click 2 points to set yours.")

    def set_line(self, a, b, frame_size):
        self.line = (a, b)
        self.line_is_default = False
        with open(STOP_LINE_FILE, "w") as f:
            json.dump({"size": list(frame_size), "line": [list(a), list(b)]}, f)
        print(f"Stop line set to {a} -> {b} and saved to {STOP_LINE_FILE}")

    def side(self, px, py):
        """Negative = behind the line (before it, for left->right traffic); >= 0 = past the line."""
        (ax, ay), (bx, by) = self.line
        dx, dy = bx - ax, by - ay
        nx, ny = dy, -dx
        if nx < 0:
            nx, ny = -nx, -ny           # make the normal point to the right (+x)
        if nx == 0:
            nx, ny = 1, 0               # horizontal line: fall back to a plain x test
        return nx * (px - ax) + ny * (py - ay)

    # ---- per frame ----
    def update(self, frame, results, names, state, now, phase_text, plate_reader):
        if state == "RED":
            if self.red_since is None:
                self.red_since = now
        else:
            self.red_since = None
        red_long_enough = state == "RED" and now - self.red_since >= RED_GRACE_SEC

        boxes = results.boxes
        if self.line is not None and boxes is not None and len(boxes) > 0:
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for box, tid in zip(boxes, ids):
                if tid is None or names[int(box.cls[0])] not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self.seen[tid] += 1

                if plate_reader is not None:
                    for (px, py, text) in plate_reader.fresh:
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            self.plate_votes.setdefault(tid, Counter())[text] += 1

                # front of a left->right vehicle = its right edge, at mid height
                s = self.side(x2, (y1 + y2) // 2)
                prev = self.prev_side.get(tid)
                self.prev_side[tid] = s

                crossed = prev is not None and prev < 0 <= s
                if (crossed and red_long_enough and tid not in self.flagged
                        and self.seen[tid] >= MIN_TRACK_FRAMES):
                    self.flag(frame, tid, names[int(box.cls[0])], (x1, y1, x2, y2),
                              now, phase_text, state)

        if len(self.prev_side) > 2000:
            self.prev_side.clear()
        self.finalize(now)

    def flag(self, frame, tid, vehicle, box, now, phase_text, state):
        self.flagged.add(tid)
        self.count += 1
        self.flash_until = now + 2.0
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        snap = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(snap, self.line[0], self.line[1], (0, 0, 255), 3)
        cv2.putText(snap, f"RED LIGHT VIOLATION  id={tid}  {stamp}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        path = f"violations/violation_{time.strftime('%Y%m%d_%H%M%S')}_id{tid}.jpg"
        cv2.imwrite(path, snap)
        self.pending[tid] = {"time": now, "stamp": stamp, "vehicle": vehicle,
                             "phase": phase_text, "state": state, "image": path}
        print(f"VIOLATION {stamp} id={tid} {vehicle} crossed on {state} ({phase_text})")

    def finalize(self, now, force=False):
        for tid in list(self.pending):
            p = self.pending[tid]
            if force or now - p["time"] >= PLATE_WAIT_SEC:
                votes = self.plate_votes.get(tid)
                plate = votes.most_common(1)[0][0] if votes else "unread"
                self.writer.writerow([p["stamp"], tid, p["vehicle"], p["phase"], p["state"],
                                      plate, p["image"]])
                self.file.flush()
                print(f"VIOLATION RECORDED id={tid} plate={plate} image={p['image']}")
                del self.pending[tid]

    def draw(self, frame, state, now, pending_line_points):
        colour = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}[state]
        if self.line is not None:
            cv2.line(frame, self.line[0], self.line[1], colour, 4)
            cv2.putText(frame, "STOP LINE" + (" (default, press L)" if self.line_is_default else ""),
                        (self.line[0][0] + 8, self.line[0][1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        for p in pending_line_points:
            cv2.circle(frame, p, 6, (255, 0, 255), -1)
        if now < self.flash_until:
            cv2.putText(frame, "RED LIGHT VIOLATION!", (20, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)

    def close(self):
        self.finalize(time.time(), force=True)
        self.file.close()


# =========================
# OBSERVATION (matches sumo-rl layout)
# =========================

def build_observation(phase, min_green_ok, counts, queued):
    """[phase one-hot (4)] + [min_green flag (1)] + [density per lane (8)] + [queue per lane (8)]"""
    phase_ids = np.zeros(N_GREEN_PHASES, dtype=np.float32)
    phase_ids[phase] = 1.0

    lanes_per_approach = {}
    for lane in SUMO_LANES:
        a = LANE_TO_APPROACH[lane[0]]
        lanes_per_approach[a] = lanes_per_approach.get(a, 0) + 1

    density, queue = [], []
    for lane in SUMO_LANES:
        a = LANE_TO_APPROACH[lane[0]]
        n = lanes_per_approach[a]
        density.append(min(1.0, counts[a] / n / LANE_CAPACITY))
        queue.append(min(1.0, queued[a] / n / LANE_CAPACITY))

    obs = np.concatenate([phase_ids, [float(min_green_ok)], density, queue]).astype(np.float32)
    assert obs.shape == (OBS_SHAPE,), obs.shape
    return obs


# =========================
# POLICY + SIGNAL STATE
# =========================

def load_policy(agent_path, algo):
    if not agent_path:
        print("No --agent given: using rule-based fallback (more cars wins).")
        return None
    from stable_baselines3 import DQN, PPO, A2C
    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    agent = cls.load(agent_path)
    print(f"Loaded {algo.upper()} agent from {agent_path}")
    return agent


def choose_action(agent, obs, counts):
    if agent is not None:
        action, _ = agent.predict(obs, deterministic=True)
        return int(action)
    ns = counts["north"] + counts["south"]
    ew = counts["east"] + counts["west"]
    return 0 if ns >= ew else 2


class Signal:
    """Mirrors sumo-rl timing: decide every DELTA_TIME, YELLOW_TIME between phases, MIN_GREEN respected."""

    def __init__(self):
        now = time.time()
        self.phase = 0
        self.next_phase = 0
        self.yellow = False
        self.yellow_start = 0.0
        self.last_change = now
        self.last_decision = now

    def min_green_ok(self, now):
        return now - self.last_change >= MIN_GREEN + YELLOW_TIME

    def step(self, now, get_action):
        if self.yellow and now - self.yellow_start >= YELLOW_TIME:
            self.phase = self.next_phase
            self.yellow = False
            self.last_change = now
        if not self.yellow and now - self.last_decision >= DELTA_TIME:
            self.last_decision = now
            action = get_action()
            if action != self.phase and self.min_green_ok(now):
                self.next_phase = action
                self.yellow = True
                self.yellow_start = now

    def text(self):
        return "YELLOW -> " + PHASE_NAMES[self.next_phase] if self.yellow else PHASE_NAMES[self.phase]


# =========================
# MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, video path, or IP camera URL")
    ap.add_argument("--yolo", default="yolov8n.pt", help="vehicle detector")
    ap.add_argument("--plates-model", default="plates.pt", help="plate detector")
    ap.add_argument("--no-plates", action="store_true", help="traffic control + violations, no plate reading")
    ap.add_argument("--debug-plates", action="store_true", help="print every OCR read")
    ap.add_argument("--min-plate-width", type=int, default=MIN_PLATE_WIDTH)
    ap.add_argument("--plate-conf", type=float, default=0.4, help="plate detector confidence")
    ap.add_argument("--agent", default=None, help="path to trained SB3 .zip")
    ap.add_argument("--algo", default="dqn", choices=["dqn", "ppo", "a2c"])
    ap.add_argument("--display-width", type=int, default=1280,
                    help="shrink the video WINDOW to this width if the video is larger (detection stays full size)")
    ap.add_argument("--process-width", type=int, default=0,
                    help="resize every frame to this width BEFORE detection (e.g. 960 or 640) = lower resolution, faster")
    ap.add_argument("--green-phases", default=DEFAULT_GREEN_PHASES,
                    help="comma list of signal phases in which THIS (left->right) approach has green")
    args = ap.parse_args()

    green_phases = {int(p) for p in args.green_phases.split(",") if p.strip()}

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    counter = VehicleCounter(YOLO(args.yolo))
    plates = None if args.no_plates else PlateReader(
        args.plates_model, debug=args.debug_plates,
        min_width=args.min_plate_width, conf=args.plate_conf)
    agent = load_policy(args.agent, args.algo)
    sig = Signal()
    viol = ViolationDetector(green_phases)

    ui = {"set_line": False, "points": [], "frame_size": None, "scale": 1.0}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = int(x / ui["scale"]), int(y / ui["scale"])      # window pixels -> real frame pixels
        print(f"click: x={x}, y={y}")
        if ui["set_line"]:
            ui["points"].append((x, y))
            if len(ui["points"]) == 2:
                viol.set_line(ui["points"][0], ui["points"][1], ui["frame_size"])
                ui["set_line"] = False
                ui["points"] = []

    win = "Traffic control + plates + red-light check (L = set stop line, ESC/q = quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("Started. Press L, then click two points on the stop line (left->right traffic crosses it).")

    last_print = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.process_width and frame.shape[1] > args.process_width:
            if ui.get("pf") is None:
                ui["pf"] = args.process_width / frame.shape[1]
                for k_, (a_, b_, c_, d_) in list(LANE_ZONES.items()):   # keep lane boxes aligned
                    LANE_ZONES[k_] = (int(a_ * ui["pf"]), int(b_ * ui["pf"]), int(c_ * ui["pf"]), int(d_ * ui["pf"]))
                print(f"Processing at lower resolution (scale {ui['pf']:.2f})")
            frame = cv2.resize(frame, None, fx=ui["pf"], fy=ui["pf"], interpolation=cv2.INTER_AREA)

        if ui["frame_size"] is None:
            ui["frame_size"] = (frame.shape[1], frame.shape[0])
            ui["scale"] = min(1.0, args.display_width / frame.shape[1])
            print(f"Video is {frame.shape[1]}x{frame.shape[0]}; window scale = {ui['scale']:.2f}")
            viol.load_or_default(ui["frame_size"])

        # ---- traffic ----
        counts, queued, results = counter.analyse(frame)
        now = time.time()

        def decide():
            obs = build_observation(sig.phase, sig.min_green_ok(now), counts, queued)
            return choose_action(agent, obs, counts)

        sig.step(now, decide)

        # ---- plates (on the raw frame, before anything is drawn) ----
        if plates is not None:
            plates.process(frame, PHASE_NAMES[sig.phase])

        # ---- red-light check (also on the raw frame) ----
        state = approach_state(sig, green_phases)
        viol.update(frame, results, counter.model.names, state, now, PHASE_NAMES[sig.phase], plates)

        # ---- drawing ----
        frame = results.plot(img=frame)
        for (x1, y1, x2, y2) in LANE_ZONES.values():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if plates is not None:
            plates.draw(frame)
        viol.draw(frame, state, now, ui["points"])

        cv2.putText(frame, sig.text(), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255) if sig.yellow else (0, 255, 0), 2)
        y = 80
        for d in LANE_ZONES:
            cv2.putText(frame, f"{d}: {counts[d]} (queued {queued[d]})", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y += 32
        cv2.putText(frame, f"Eastbound light: {state}   Violations: {viol.count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if ui["set_line"]:
            cv2.putText(frame, f"Click stop-line point {len(ui['points']) + 1} of 2", (20, y + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        shown = frame if ui["scale"] >= 1.0 else cv2.resize(
            frame, None, fx=ui["scale"], fy=ui["scale"], interpolation=cv2.INTER_AREA)
        cv2.imshow(win, shown)

        if now - last_print >= 1.0:
            last_print = now
            print(f"counts={counts} queued={queued} signal={sig.text()} eastbound={state}")

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k in (ord("l"), ord("L")):
            ui["set_line"] = True
            ui["points"] = []
            print("Click two points on the stop line.")

    cap.release()
    viol.close()
    print(f"Saved violations to {VIOLATION_LOG} ({viol.count} flagged)")
    if plates is not None:
        plates.close()
        print(f"Saved plate log to {PLATE_LOG}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
