#exports frames from a video using YOLOv8
# This script uses the YOLOv8 model to extract frames from a video file based on detected classes.
# It allows the user to select a video file, a YOLOv8 model, and a specific class to filter the frames.
# The extracted frames are saved in a specified directory with a timestamp and class name in the filename.
# can save a maximun number of frames 
#use an algorithm to select the frames to save by selectiong the frames that are most distant from each othe

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import threading
import math

class VideoFrameExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Class Episode Extractor")

        # Variables
        self.video_paths = []
        self.model_path = ""
        self.model = None
        self.class_list = []
        self.selected_class = tk.StringVar()
        self.frame_step = tk.IntVar(value=10)
        self.max_images = tk.IntVar(value=500)
        self.stopped = False
        self.resume_from_frame = 0

        # UI
        self.setup_ui()

    def setup_ui(self):
        tk.Button(self.root, text="Select Video File", command=self.select_video).pack(pady=5)
        tk.Button(self.root, text="Select YOLOv8 Model", command=self.select_model).pack(pady=5)

        tk.Label(self.root, text="Select Class:").pack()
        self.class_dropdown = ttk.Combobox(self.root, textvariable=self.selected_class, state="readonly")
        self.class_dropdown.pack(pady=5)

        tk.Label(self.root, text="Frame Step:").pack()
        tk.Entry(self.root, textvariable=self.frame_step).pack(pady=5)

        tk.Label(self.root, text="Max Frames to Save:").pack()
        tk.Entry(self.root, textvariable=self.max_images).pack(pady=5)

        self.progress_label = tk.Label(self.root, text="Progress: 0/0")
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=5)

        self.start_button = tk.Button(self.root, text="Start", command=self.start_processing_thread)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_processing)
        self.stop_button.pack(pady=5)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if path:
            self.video_paths = [path]

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLOv8 model", "*.pt")])
        if path:
            self.model_path = path
            self.model = YOLO(self.model_path)
            self.class_list = self.model.names
            self.class_dropdown['values'] = list(self.class_list.values())
            if self.class_list:
                self.class_dropdown.current(0)

    def stop_processing(self):
        self.stopped = True

    def start_processing_thread(self):
        self.stopped = False
        thread = threading.Thread(target=self.process_video)
        thread.start()

    def process_video(self):
        if not self.video_paths or not self.model:
            messagebox.showerror("Error", "Please select a video file and a YOLO model.")
            return

        class_name = self.selected_class.get()
        if not class_name:
            messagebox.showerror("Error", "Please select a class.")
            return

        step = self.frame_step.get()
        max_to_save = self.max_images.get()

        video_path = self.video_paths[0]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(video_dir, class_name)
        os.makedirs(output_dir, exist_ok=True)

        self.progress_bar["maximum"] = total_frames

        detected_frame_indices = []
        class_episodes = []
        current_episode = []

        frame_index = self.resume_from_frame

        while frame_index < total_frames and not self.stopped:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            boxes = results.boxes
            names = results.names

            if boxes is not None:
                detected_classes = [names[int(cls)] for cls in boxes.cls]
                if class_name in detected_classes:
                    detected_frame_indices.append((frame_index, frame.copy()))
                    if not current_episode:
                        current_episode = [(frame_index, frame.copy())]  # first frame
                    elif frame_index - current_episode[-1][0] <= 60:
                        current_episode.append((frame_index, frame.copy()))
                    else:
                        if len(current_episode) >= 2:
                            class_episodes.append((current_episode[0], current_episode[-1]))
                        current_episode = [(frame_index, frame.copy())]

            frame_index += step
            self.progress_bar["value"] = min(frame_index, total_frames)
            self.progress_label.config(text=f"Progress: {min(frame_index, total_frames)}/{total_frames}")
            self.root.update_idletasks()

        cap.release()
        self.resume_from_frame = frame_index

        if current_episode and len(current_episode) >= 2:
            class_episodes.append((current_episode[0], current_episode[-1]))

        # Flatten and count all saved detection frames
        all_detected = [ep[0] for ep in class_episodes] + [ep[1] for ep in class_episodes]

        if len(all_detected) > max_to_save:
            interval = len(all_detected) / max_to_save
            indices = [int(i * interval) for i in range(max_to_save)]
            selected_frames = [all_detected[i] for i in indices]
        else:
            selected_frames = all_detected

        for idx, (f_idx, frame) in enumerate(selected_frames):
            timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
            filename = f"{video_name}_{class_name}_{timestamp}_{f_idx}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

        messagebox.showinfo("Done", f"Exported {len(selected_frames)} frames from class episodes.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFrameExtractorApp(root)
    root.mainloop()