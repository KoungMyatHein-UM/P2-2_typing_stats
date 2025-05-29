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
    """
    Normalize a given key name and return it unchanged. This function takes a string
    and simply returns it without performing any transformation or processing.

    :param key_name: The key name to be normalized.
    :type key_name: str
    :return: The unchanged key name.
    :rtype: str
    """
    return key_name

def wrap(sym):
    """
    Wraps a given symbol with angle brackets.

    This function takes a single symbol and returns the symbol enclosed
    within angle brackets ('<', '>'). It is commonly used for formatting
    or representing elements with specific delimiters.

    :param sym: The symbol to be wrapped within angle brackets
    :type sym: str
    :return: A string where the input symbol is wrapped with angle brackets
    :rtype: str
    """
    return f"<{sym}>"

# — core handlers —
def on_key_down(event):
    """
    Handles the key-down event and updates timing data for flight times, n-gram
    times, and dwell start times. The method processes normalized key inputs and
    calculates timing for key events based on previous key press timestamps.

    :param event: The key event object containing information about the key press.
    :type event: Any
    :return: None
    """
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
    """
    Handles the event triggered when a key is released. Captures the key's dwell
    time (duration it was held down) and updates relevant global variables for
    processing dwell times and recording the key release event.

    This function processes the timing of when a specific key is released, computes
    the dwell time from when the key was first pressed, and stores this information
    for further analysis. Additionally, it updates the global variables to track
    the last key released and the time of release.

    :param event: The key event containing details about the release action of a
                  specific key, such as its name.
    :type event: Any
    """
    global last_keyup_stamp, last_keyup_key
    key = norm(event.name); now = time.perf_counter() * 1000
    if key in dwell_starts:
        dwell = now - dwell_starts.pop(key)
        dwell_times[wrap(key)].append(int(dwell))
    last_keyup_stamp = now
    last_keyup_key = key

# — saving logic —
def save_data(clear_buffers=True):
    """
    Saves typing timing data to a JSON file and optionally clears the internal buffers
    after saving. The function generates a timestamped filename based on the current
    datetime, serializes the timing data stored in `dwell_times`, `flight_times`, and
    `ngram_times` dictionaries, and writes it to the generated file in a JSON format.
    The optional clearing of buffers allows the usage of this function as part of a
    buffer resetting mechanism after persisting data.

    :param clear_buffers: A boolean parameter that determines whether the `dwell_times`,
                          `flight_times`, and `ngram_times` dictionaries should be cleared
                          once the data is successfully saved. Defaults to True.
    :return: None
    """
    dir = "data"

    if not os.path.exists(dir):
        os.makedirs(dir)

    base, ext = "typing-timings", ".json"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{dir}/{base}-{ts}{ext}"
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
    """
    Handles termination signals to gracefully save data and exit the program.

    This function is designed to handle the termination signals (such as SIGTERM or
    SIGINT). Upon receiving the signal, any unsaved data is saved, and the program
    is terminated gracefully. This ensures that data consistency and integrity are
    maintained even during abrupt terminations.

    :param signum: The signal number that was received indicating termination.
    :type signum: int
    :param frame: The current stack frame (as provided by the signal handler).
    :type frame: FrameType
    :return: None
    """
    global running
    save_data(clear_buffers=False)
    print("Cheers! Exiting.")
    running = False

def main():
    """
    Captures global keystrokes, processes them, and manages saving the data at regular intervals
    or when the program exits. It listens for key press and release events to track user input.
    The program also automatically saves tracked data hourly for safety and persistency.

    :param running: A flag to control the main loop execution and allow graceful termination.
    :type running: bool

    :raises KeyboardInterrupt: Triggered when the user presses Ctrl+C to save data and stop
                               the application.

    :return: None
    """
    print("Capturing global keystrokes… press Ctrl+C to stop and save. Auto saves every hour.")
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
