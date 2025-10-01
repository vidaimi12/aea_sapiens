# Mediapipe based Hand Tracker

First download the model (hand_landmarker.task) from this site: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

### Standard Libraries
| Library   | Submodules |
|-----------|------------|
| `warnings` | – |
| `os`       | – |
| `logging`  | – |
| `sys`      | – |
| `csv`      | – |
| `time`     | – |
| `json`     | – |
| `tkinter`  | `filedialog`, `ttk`, `messagebox` |

### Third-party Libraries
| Library   | Submodules |
|-----------|------------|
| `cv2` (OpenCV) | – |
| `mediapipe` | `tasks`, `tasks.python`, `tasks.python.vision` |
| `numpy` | – |

# How to use:
First code asks you to select a camera, and a location to save the output csv
that is in this format:
 
Each row corresponds to **one hand landmark** detected in a frame.

---

## **Columns**

| Column Name         | Type     | Description |
|---------------------|----------|-------------|
| `timestamp`         | float    | Unix timestamp (seconds) when the frame was processed. |
| `hand_idx`          | int      | Index of the detected hand in the frame (0 for first hand, 1 for second hand, etc.). |
| `handedness`        | string   | Predicted hand label (`Left` or `Right`). |
| `landmark_idx`      | int      | Landmark index (0–20) corresponding to the MediaPipe hand landmarks. |
| `x`                 | float    | Normalized x-coordinate of the landmark (0.0–1.0, relative to image width). |
| `y`                 | float    | Normalized y-coordinate of the landmark (0.0–1.0, relative to image height). |
| `z`                 | float    | Normalized depth of the landmark (negative = closer to the camera). |
| `thumb_ext_angle`   | float    | Extension angle (degrees) of the thumb relative to the wrist and MCP joint. |
| `index_ext_angle`   | float    | Extension angle (degrees) of the index finger relative to the wrist and MCP joint. |
| `middle_ext_angle`  | float    | Extension angle (degrees) of the middle finger relative to the wrist and MCP joint. |
| `ring_ext_angle`    | float    | Extension angle (degrees) of the ring finger relative to the wrist and MCP joint. |
| `pinky_ext_angle`   | float    | Extension angle (degrees) of the pinky finger relative to the wrist and MCP joint. |
| `hand_state`        | string   | Hand state classification (`Open`, `Closed`, `Unknown`, `Error`). |

---

## **Notes**
- The **first landmark row per hand** (landmark `0`, the wrist) contains the extension angles and `hand_state`.  
- For all other landmarks of the same hand, these fields are left **empty** to avoid duplication.
- One frame with two hands produces **42 rows** (21 landmarks × 2 hands).
- The **FPS** is shown only on the video window, not saved in the CSV.

Then the preview image is shown and it is possible to exit and save the output by pressing the q button.
