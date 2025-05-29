#!/usr/bin/env python3
import time
import json
import signal
import os
from datetime import datetime
from collections import defaultdict, deque

try:
    import keyboard   # pip install keyboard
except ImportError:
    print("Please install the `keyboard` library: pip install keyboard")
    exit(1)

# — tunables (in milliseconds) —
MAX_MS    = 1500
GAP_MS    = 1000
CONTEXT_K = 5

# — data stores —
dwell_times   = defaultdict(list)
flight_times  = defaultdict(list)
ngram_times   = defaultdict(list)

# — for tracking —
dwell_starts     = {}
last_keyup_stamp = None
last_keyup_key   = None

prev_chars = deque(maxlen=CONTEXT_K)
prev_stamp = None

running = True

def norm(key_name):
    return key_name

def wrap(sym):
    return f"<{sym}>"

# — core handlers —
def on_key_down(event):
    global last_keyup_stamp, last_keyup_key, prev_stamp
    key = norm(event.name); now = time.perf_counter() * 1000
    wrapped = wrap(key)

    # flight
    if last_keyup_stamp is not None:
        prev_wrapped = wrap(last_keyup_key)
        fk = f"[{prev_wrapped}]->[{wrapped}]"
        flight_times[fk].append(int(now - last_keyup_stamp))

    # n-gram
    if prev_stamp is not None and prev_chars:
        gap = now - prev_stamp
        if gap < GAP_MS and gap <= MAX_MS:
            for j in range(1, len(prev_chars) + 1):
                ctx = list(prev_chars)[-j:]
                ctx_wrapped = ''.join(wrap(c) for c in ctx)
                pk = f"[{ctx_wrapped}]->[{wrapped}]"
                ngram_times[pk].append(int(gap))

    # dwell start
    dwell_starts[key] = now
    prev_chars.append(key)
    prev_stamp = now

def on_key_up(event):
    global last_keyup_stamp, last_keyup_key
    key = norm(event.name); now = time.perf_counter() * 1000
    if key in dwell_starts:
        dwell = now - dwell_starts.pop(key)
        dwell_times[wrap(key)].append(int(dwell))
    last_keyup_stamp = now
    last_keyup_key = key

# — saving logic —
def save_data(clear_buffers=True):
    """Dump current timings to a timestamped file; optionally clear."""
    base, ext = "typing-timings", ".json"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{base}-{ts}{ext}"
    out = {
        "dwell_times":  dict(dwell_times),
        "flight_times": dict(flight_times),
        "ngram_times":  dict(ngram_times),
    }
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved to {fname}]")

    if clear_buffers:
        dwell_times.clear()
        flight_times.clear()
        ngram_times.clear()

def save_and_exit(signum, frame):
    """SIGINT handler: save one last time and quit."""
    global running
    save_data(clear_buffers=False)
    print("Cheers! Exiting.")
    running = False

def main():
    print("Capturing global keystrokes… press Ctrl+C to stop and save.")
    keyboard.on_press(on_key_down)
    keyboard.on_release(on_key_up)
    signal.signal(signal.SIGINT, save_and_exit)

    last_hourly = time.time()
    while running:
        time.sleep(0.1)
        if time.time() - last_hourly >= 3600:
            save_data(clear_buffers=True)
            last_hourly = time.time()

if __name__ == "__main__":
    main()
