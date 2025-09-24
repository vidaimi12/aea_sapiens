import warnings
import os
import logging
import sys

# Configure warnings and logging
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import csv
import time
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Assuming 'utilities.py' is in the same directory and contains draw_landmarks_on_image
try:
    from utilities import draw_landmarks_on_image
except ImportError:
    print("WARNING: 'utilities.py' not found or 'draw_landmarks_on_image' function is missing.")
    print("         Landmarks will not be drawn visually if the utility is absent.")
    # Fallback: create a dummy function if not found
    def draw_landmarks_on_image(rgb_image, detection_result):
        # This function should draw landmarks. If utilities.py is missing,
        # it will just return the image without drawing.
        return rgb_image

# --- Constants for Hand Landmarks ---
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20
# --- End Constants ---

def get_settings_path():
    """Returns the absolute path to the settings file"""
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(application_path, 'settings.txt')

def load_settings():
    """Load settings from file"""
    default_settings = {'camera_index': 0, 'model_path': 'hand_landmarker.task'}
    settings_path = get_settings_path()
    try:
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                loaded_settings = json.load(f)
                for key in default_settings:
                    if key not in loaded_settings:
                        loaded_settings[key] = default_settings[key]
                return loaded_settings
    except Exception as e:
        print(f"Error loading settings: {e}. Using defaults.")
    return default_settings

def save_settings(settings):
    """Save settings to file"""
    settings_path = get_settings_path()
    try:
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
    except Exception as e:
        print(f"Error saving settings: {e}")

def get_available_cameras():
    """Returns a list of available camera indices and their names"""
    camera_list = []
    index = 0
    print("Searching for available cameras...")
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if not cap.isOpened():
            if index < 5:
                cap.release()
                index += 1
                continue
            break
        ret, _ = cap.read()
        if ret:
            camera_name = f"Camera {index}"
            try:
                backend_name = cap.getBackendName()
                if backend_name and backend_name != "CAP_GSTREAMER":
                     camera_name = f"Camera {index} ({backend_name})"
            except:
                pass
            camera_list.append((index, camera_name))
            print(f"  Found: {camera_name}")
        cap.release()
        index += 1
        if index > 10:
            break
    if not camera_list:
        print("No cameras found.")
    return camera_list

def locate_model_file():
    """Shows file dialog to locate the model file"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Locate hand_landmarker.task file",
        filetypes=[("Task files", "*.task"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path if file_path else 'hand_landmarker.task'

def select_camera():
    """Shows camera selection dialog and returns selected camera index"""
    settings = load_settings()
    root = tk.Tk()
    root.title("Camera Selection")
    camera_var = tk.StringVar()
    cameras = get_available_cameras()
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    ttk.Label(frame, text="Select Camera:").grid(row=0, column=0, pady=5, sticky=tk.W)
    combo = ttk.Combobox(frame, textvariable=camera_var, width=40, state="readonly")
    if cameras:
        combo['values'] = [f"{idx}: {name}" for idx, name in cameras]
        default_cam_idx_val = settings.get('camera_index', 0)
        current_selection_str = None
        for val_str in combo['values']:
            if val_str.startswith(f"{default_cam_idx_val}:"):
                current_selection_str = val_str
                break
        if current_selection_str:
             combo.set(current_selection_str)
        elif combo['values']:
            combo.set(combo['values'][0])
    else:
        combo.set("No cameras found")
        combo['values'] = ["No cameras found"]
    combo.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
    selected_idx_val = tk.IntVar(value=settings.get('camera_index', 0))
    def on_select_confirm():
        if combo.get() != "No cameras found" and combo.get():
            try:
                selected_idx_val.set(int(combo.get().split(':')[0]))
            except ValueError:
                selected_idx_val.set(0)
        else:
            if not cameras:
                 messagebox.showerror("Camera Error", "No cameras detected.")
                 selected_idx_val.set(-1)
            else:
                selected_idx_val.set(0)
        root.quit()
        root.destroy()
    ttk.Button(frame, text="Select", command=on_select_confirm).grid(row=2, column=0, pady=10)
    root.protocol("WM_DELETE_WINDOW", on_select_confirm)
    root.mainloop()
    final_selected_idx = selected_idx_val.get()
    if final_selected_idx != -1:
        settings['camera_index'] = final_selected_idx
        save_settings(settings)
    return final_selected_idx

def get_save_filename():
    """Shows save file dialog and returns selected path"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title="Save hand tracking data as"
    )
    root.destroy()
    return file_path if file_path else None

def calculate_extension_angle(wrist_lm, finger_base_lm, finger_tip_lm):
    """
    Calculates the finger extension angle.
    Angle between vector (finger_tip - finger_base) and (wrist - finger_base).
    """
    # Convert points to numpy arrays
    w = np.array([wrist_lm.x, wrist_lm.y, wrist_lm.z])
    b = np.array([finger_base_lm.x, finger_base_lm.y, finger_base_lm.z])
    t = np.array([finger_tip_lm.x, finger_tip_lm.y, finger_tip_lm.z])

    # Create vectors
    v_tip_base = t - b  # Vector from finger base to finger tip
    v_wrist_base = w - b # Vector from finger base to wrist

    # Calculate dot product and magnitudes
    dot_product = np.dot(v_tip_base, v_wrist_base)
    magnitude_v_tip_base = np.linalg.norm(v_tip_base)
    magnitude_v_wrist_base = np.linalg.norm(v_wrist_base)

    # Calculate cosine of the angle
    # Add epsilon to avoid division by zero
    cosine_angle = dot_product / (magnitude_v_tip_base * magnitude_v_wrist_base + 1e-6)

    # Clip to avoid domain errors with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def recognize_hand_state_from_extension(extension_angles):
    """
    Recognizes hand state ("Open" or "Closed") based on finger extension angles.
    extension_angles: [thumb, index, middle, ring, pinky]
    """
    if not extension_angles or len(extension_angles) < 5:
        return "Unknown"

    # Using index, middle, ring, pinky for the "open/closed" state
    # as per the provided pandas script logic.
    # angle > 90 means the finger is extended away from the wrist-base line.
    # angle < 90 means the finger is flexed towards the wrist-base line.
    # The interpretation of "open" in the pandas script is when these angles are large.
    # However, the vectors are Tip-Base and Wrist-Base.
    # If finger is straight along palm, Tip-Base and Wrist-Base point in similar directions from Base, angle small.
    # If finger is extended "outwards", angle will be larger.
    # The pandas example uses `if angle <= 90: all_above_90 = False`. So "open" means all angles > 90.

    index_ext, middle_ext, ring_ext, pinky_ext = extension_angles[1], extension_angles[2], extension_angles[3], extension_angles[4]
    
    # Threshold based on the pandas script's logic: angle > 90 for "open"
    # This implies that "open" means the fingers are splayed out significantly
    # from the line connecting their base to the wrist.
    # Let's stick to the provided logic:
    threshold = 90.0 
    
    if (index_ext > threshold and
        middle_ext > threshold and
        ring_ext > threshold and
        pinky_ext > threshold):
        return "Open"
    else:
        return "Closed"

def display_info_panel(frame, hand_idx, angles, hand_state_name, start_y):
    """Displays hand state and extension angle info on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_state = 0.7
    font_scale_angle = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White text
    shadow_color = (0,0,0) # Black shadow
    bg_opacity = 0.6
    
    current_y = start_y
    x_pos = 10
    line_height_state = 30
    line_height_angle = 20
    padding = 5

    overlay = frame.copy()

    state_text = f"Hand {hand_idx}: {hand_state_name}"
    (w_s, h_s), _ = cv2.getTextSize(state_text, font, font_scale_state, font_thickness)
    
    cv2.rectangle(overlay, (x_pos - padding, current_y - h_s - padding), 
                  (x_pos + w_s + padding, current_y + padding + 3), (50, 50, 50), -1)
    cv2.putText(overlay, state_text, (x_pos + 1, current_y + 1), font, font_scale_state, shadow_color, font_thickness, cv2.LINE_AA)
    cv2.putText(overlay, state_text, (x_pos, current_y), font, font_scale_state, text_color, font_thickness, cv2.LINE_AA)
    current_y += line_height_state

    angle_labels = ["Th:", "Idx:", "Mid:", "Rng:", "Pky:"] # Extension angles
    if angles and len(angles) == 5:
        for i, label in enumerate(angle_labels):
            angle_text = f"{label} {angles[i]:.0f} deg"
            (w_a, h_a), _ = cv2.getTextSize(angle_text, font, font_scale_angle, font_thickness)
            
            cv2.rectangle(overlay, (x_pos + 5 - padding, current_y - h_a - padding), 
                          (x_pos + 5 + w_a + padding, current_y + padding + 2), (70, 70, 70), -1)
            cv2.putText(overlay, angle_text, (x_pos + 5 + 1, current_y + 1), font, font_scale_angle, shadow_color, font_thickness, cv2.LINE_AA)
            cv2.putText(overlay, angle_text, (x_pos + 5, current_y), font, font_scale_angle, text_color, font_thickness, cv2.LINE_AA)
            current_y += line_height_angle
    
    cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)
    return current_y

def main():
    camera = None 
    try:
        settings = load_settings()
        model_path = settings.get('model_path', 'hand_landmarker.task')
        if not os.path.exists(model_path):
            print(f"Model file '{model_path}' not found. Please locate it.")
            model_path = locate_model_file()
            if not model_path or not os.path.exists(model_path):
                messagebox.showerror("Error", "No valid model file selected. Exiting.")
                return
            settings['model_path'] = model_path
            save_settings(settings)

        camera_idx = select_camera()
        if camera_idx == -1:
             print("No camera selected or available. Exiting.")
             return

        save_file = get_save_filename()
        if not save_file: 
            print("No save file selected. Exiting.")
            return

        camera = cv2.VideoCapture(camera_idx)
        if not camera.isOpened():
            backend_name = camera.getBackendName() if hasattr(camera, 'getBackendName') else "Unknown backend"
            raise RuntimeError(f"Failed to open camera {camera_idx} (Backend: {backend_name}).")
    
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, # Set running mode for video
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        detector = vision.HandLandmarker.create_from_options(options)

        cv2.namedWindow('Hand Landmarks', cv2.WINDOW_NORMAL)
        print(f"Starting hand tracking. Press 'q' or 'ESC' to quit.")
        print(f"Saving data to: {save_file}")

        with open(save_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Updated CSV Header
            header = ['timestamp', 'hand_idx', 'handedness', 'landmark_idx', 'x', 'y', 'z',
                      'thumb_ext_angle', 'index_ext_angle', 'middle_ext_angle', 
                      'ring_ext_angle', 'pinky_ext_angle', 'hand_state']
            csv_writer.writerow(header)

            frame_count = 0
            start_process_time = time.time()

            while camera.isOpened():
                success, frame = camera.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    time.sleep(0.1) 
                    continue
                
                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                current_time_ms = int(time.time() * 1000) 
                detection_result = detector.detect_for_video(mp_image, current_time_ms)
                
                timestamp_csv = time.time() 
                annotated_frame = frame.copy() 

                if detection_result.hand_landmarks:
                    annotated_frame = draw_landmarks_on_image(annotated_frame, detection_result)
                
                info_display_y_start = 25 

                if detection_result.hand_landmarks:
                    for hand_idx, hand_landmarks_list in enumerate(detection_result.hand_landmarks):
                        handedness_name = "Unknown"
                        if detection_result.handedness and hand_idx < len(detection_result.handedness):
                            handedness_name = detection_result.handedness[hand_idx][0].category_name
                        
                        current_hand_ext_angles = []
                        hand_state = "Unknown"
                        
                        if len(hand_landmarks_list) == 21: # Ensure all landmarks are present
                            landmarks = hand_landmarks_list
                            wrist_lm = landmarks[WRIST]

                            try:
                                # Thumb Extension Angle (MCP as base)
                                thumb_ext_angle = calculate_extension_angle(wrist_lm, landmarks[THUMB_MCP], landmarks[THUMB_TIP])
                                current_hand_ext_angles.append(thumb_ext_angle)

                                # Index Finger Extension Angle
                                index_ext_angle = calculate_extension_angle(wrist_lm, landmarks[INDEX_FINGER_MCP], landmarks[INDEX_FINGER_TIP])
                                current_hand_ext_angles.append(index_ext_angle)

                                # Middle Finger Extension Angle
                                middle_ext_angle = calculate_extension_angle(wrist_lm, landmarks[MIDDLE_FINGER_MCP], landmarks[MIDDLE_FINGER_TIP])
                                current_hand_ext_angles.append(middle_ext_angle)

                                # Ring Finger Extension Angle
                                ring_ext_angle = calculate_extension_angle(wrist_lm, landmarks[RING_FINGER_MCP], landmarks[RING_FINGER_TIP])
                                current_hand_ext_angles.append(ring_ext_angle)

                                # Pinky Finger Extension Angle
                                pinky_ext_angle = calculate_extension_angle(wrist_lm, landmarks[PINKY_MCP], landmarks[PINKY_TIP])
                                current_hand_ext_angles.append(pinky_ext_angle)
                                
                                hand_state = recognize_hand_state_from_extension(current_hand_ext_angles)

                            except Exception as e:
                                print(f"Error calculating extension angles/state for hand {hand_idx}: {e}")
                                current_hand_ext_angles = [0,0,0,0,0] 
                                hand_state = "Calc Error"
                        else:
                            print(f"Hand {hand_idx} does not have 21 landmarks. Skipping angle calculation.")
                            current_hand_ext_angles = [0,0,0,0,0]
                            hand_state = "Landmark Error"


                        info_display_y_start = display_info_panel(annotated_frame, hand_idx, 
                                                                    current_hand_ext_angles, hand_state, 
                                                                    info_display_y_start)
                        info_display_y_start += 15 

                        # Write landmark data and once per hand, the calculated angles and state
                        for lm_idx, landmark in enumerate(hand_landmarks_list):
                            row_data = [timestamp_csv, hand_idx, handedness_name, lm_idx, 
                                        landmark.x, landmark.y, landmark.z]
                            if lm_idx == 0: # Add angles and state only for the first landmark of this hand
                                row_data.extend([f"{angle:.2f}" for angle in current_hand_ext_angles])
                                row_data.append(hand_state)
                            else: # Fill with empty strings for subsequent landmarks
                                row_data.extend([""] * 6) # 5 angles + 1 state
                            csv_writer.writerow(row_data)
                
                end_process_time = time.time()
                fps = frame_count / (end_process_time - start_process_time + 1e-6)
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (annotated_frame.shape[1] - 150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Hand Landmarks', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
    except RuntimeError as e:
        print(f"A runtime error occurred: {e}")
        if tk._default_root: tk.Tk().withdraw() # Hide main Tk window if it exists
        messagebox.showerror("Runtime Error", str(e))
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())
        if tk._default_root: tk.Tk().withdraw()
        messagebox.showerror("Unexpected Error", f"{e}\n\n{traceback.format_exc()}")
    finally:
        if camera and camera.isOpened():
            print("Releasing camera...")
            camera.release()
        print("Closing windows...")
        cv2.destroyAllWindows()
        print("Exiting.")

if __name__ == "__main__":
    main()