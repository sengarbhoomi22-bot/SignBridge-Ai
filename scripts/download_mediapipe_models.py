import urllib.request
import os

os.makedirs("data/raw/isl-csltr/models", exist_ok=True)

models = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

for name, url in models.items():
    dest = os.path.join("data", "raw", "isl-csltr", "models", name)
    if not os.path.exists(dest):
        print(f"Downloading {name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as res, open(dest, "wb") as out:
            data = res.read()
            out.write(data)
        print(f"Saved {name} ({len(data)} bytes)")
    else:
        print(f"Already exists: {name}")

