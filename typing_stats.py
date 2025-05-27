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
MAX_MS    = 1500    # hard cap: ignore any inter-keystroke gap > this
GAP_MS    = 1000    # burst split: drop n-gram delays ≥ this
CONTEXT_K = 5       # max n-gram context length

# — data stores —
dwell_times   = defaultdict(list)  # "<key>" -> [dwell1, dwell2, …]
flight_times  = defaultdict(list)  # "[<a>]->[<b>]" -> [flight1, …]
ngram_times   = defaultdict(list)  # "[<ctx>]->[<key>]" -> [delay1, …]

# — for tracking —
dwell_starts     = {}      # raw_key -> timestamp of key-down
last_keyup_stamp = None
last_keyup_key   = None    # raw_key

prev_chars = deque(maxlen=CONTEXT_K)  # raw_key queue
prev_stamp = None

running = True

# normalise names to match your JS map
KEY_MAP = {
    'space': '␣', 'enter': '\n', 'backspace': '␈',
    'tab': '⇥', 'esc': '⎋',
    'left': '←', 'right': '→', 'up': '↑', 'down': '↓',
    'home': '⇱', 'end': '⇲',
    'page up': '⇞', 'page down': '⇟',
    'delete': '⌦', 'insert': '⎀'
}

def norm(key_name):
    """Map keyboard event.name to our atomic symbol."""
    # return KEY_MAP.get(key_name.lower(), key_name.lower())
    return key_name

def wrap(sym):
    """Encapsulate an atomic symbol in angle brackets."""
    return f"<{sym}>"

def on_key_down(event):
    global last_keyup_stamp, last_keyup_key, prev_stamp

    key = norm(event.name)              # raw atomic symbol
    wrapped = wrap(key)
    now = time.perf_counter() * 1000    # ms

    # — flight time —
    if last_keyup_stamp is not None and last_keyup_key is not None:
        prev_wrapped = wrap(last_keyup_key)
        fk = f"[{prev_wrapped}]->[{wrapped}]"
        flight_times[fk].append(int(now - last_keyup_stamp))

    # — n-gram delays —
    if prev_stamp is not None and prev_chars:
        gap = now - prev_stamp
        if gap < GAP_MS and gap <= MAX_MS:
            # for each possible context length
            for j in range(1, len(prev_chars) + 1):
                ctx_slice = list(prev_chars)[-j:]
                ctx_wrapped = ''.join(wrap(c) for c in ctx_slice)
                pk = f"[{ctx_wrapped}]->[{wrapped}]"
                ngram_times[pk].append(int(gap))

    # — start dwell timing —
    dwell_starts[key] = now

    # — update buffer & stamp —
    prev_chars.append(key)
    prev_stamp = now

def on_key_up(event):
    global last_keyup_stamp, last_keyup_key

    key = norm(event.name)
    wrapped = wrap(key)
    now = time.perf_counter() * 1000    # ms

    # — record dwell time —
    if key in dwell_starts:
        dwell = now - dwell_starts.pop(key)
        dwell_times[wrapped].append(int(dwell))

    # — prepare for next flight —
    last_keyup_stamp = now
    last_keyup_key   = key

def save_and_exit(signum, frame):
    """On Ctrl+C, dump JSON (avoiding overwrite by adding timestamp)."""
    global running

    base = "typing-timings"
    ext  = ".json"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{base}-{ts}{ext}"

    out = {
        "dwell_times":  dict(dwell_times),
        "flight_times": dict(flight_times),
        "ngram_times":  dict(ngram_times),
    }
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved timing data to {fname}. Cheers!")
    running = False

def main():
    print("Capturing global keystrokes… press Ctrl+C to stop and save.")
    keyboard.on_press(on_key_down)
    keyboard.on_release(on_key_up)
    signal.signal(signal.SIGINT, save_and_exit)

    while running:
        time.sleep(0.1)

if __name__ == "__main__":
    main()
