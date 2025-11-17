from rallyrobopilot import prepare_game_app, RemoteController
from rallyrobopilot.car import Car, JsonInputReplayer
from flask import Flask, request, jsonify
from threading import Thread
import time
import os

# Setup Flask

FPS = 25
frame_time = 1.0 / FPS

flask_app = Flask(__name__)
flask_thread = Thread(target=flask_app.run, kwargs={"host": "0.0.0.0", "port": 5000})
print("Flask server running on port 5000")
flask_thread.start()

app, car, _ = prepare_game_app("SimpleTrack")
remote_controller = RemoteController(car=car, connection_port=7654, flask_app=flask_app)
# app.run()


def get_latest_recording(directory="input_recordings"):
    """Return full path to the latest inputs_*.json file in directory, or None if none exists."""
    if not os.path.isdir(directory):
        print(f"[main] Directory '{directory}' does not exist.")
        return None

    candidates = [
        f
        for f in os.listdir(directory)
        if f.startswith("inputs_") and f.endswith(".json")
    ]
    if not candidates:
        print(f"[main] No inputs_*.json files found in '{directory}'.")
        return None

    candidates.sort()
    latest = candidates[-1]
    full_path = os.path.join(directory, latest)
    print(f"[main] Latest recording detected: {full_path}")
    return full_path


replayer = None
replayer_toggle_down = False
recording = get_latest_recording

while True:
    start_time = time.time()
    app.step()
    elapsed = time.time() - start_time
    sleep_time = frame_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
