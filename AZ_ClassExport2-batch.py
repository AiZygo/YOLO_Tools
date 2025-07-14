# This script provides a GUI for batch processing videos to extract frames where specific classes are detected.
# It uses the YOLOv8 model for object detection and allows users to select classes,
# set frame steps, and limit the number of images saved per class.
# The extracted frames are saved in a structured directory format based on the detected class and video name.
# The script also supports resuming from a specific frame index if interrupted.
# subfolders are created for each class, and images are saved with timestamps.
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import threading
import math
from collections import defaultdict

class VideoFrameExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Class Episode Extractor")

        # Variables
        self.video_paths = []
        self.model_path = ""
        self.model = None
        self.class_list = []
        self.selected_classes = []
        self.frame_step = tk.IntVar(value=10)
        self.max_images = tk.IntVar(value=500)
        self.stopped = False
        self.resume_from_frame = 0

        # UI
        self.setup_ui()

    def setup_ui(self):
        tk.Button(self.root, text="Select Video Files", command=self.select_videos).pack(pady=5)
        tk.Button(self.root, text="Select YOLOv8 Model", command=self.select_model).pack(pady=5)

        tk.Label(self.root, text="Select Classes:").pack()
        self.class_listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE, exportselection=False, height=10)
        self.class_listbox.pack(pady=5, fill=tk.X)

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

    def select_videos(self):
        paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4 *.avi")])
        if paths:
            self.video_paths = list(paths)

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLOv8 model", "*.pt")])
        if path:
            self.model_path = path
            self.model = YOLO(self.model_path)
            self.class_list = self.model.names
            self.class_listbox.delete(0, tk.END)
            for name in self.class_list.values():
                self.class_listbox.insert(tk.END, name)

    def stop_processing(self):
        self.stopped = True

    def start_processing_thread(self):
        self.stopped = False
        thread = threading.Thread(target=self.process_videos)
        thread.start()

    def get_selected_classes(self):
        selected_indices = self.class_listbox.curselection()
        return [self.class_list[i] for i in selected_indices]

    def process_videos(self):
        if not self.video_paths or not self.model:
            messagebox.showerror("Error", "Please select video files and a YOLO model.")
            return

        selected_classes = self.get_selected_classes()
        if not selected_classes:
            messagebox.showerror("Error", "Please select at least one class.")
            return

        step = self.frame_step.get()
        max_to_save = self.max_images.get()

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.progress_bar["maximum"] = total_frames

            frame_index = self.resume_from_frame
            detected_by_class = {cls: [] for cls in selected_classes}
            episodes_by_class = {cls: [] for cls in selected_classes}
            current_episodes = {cls: [] for cls in selected_classes}
            export_counts = defaultdict(int)

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
                    for cls in selected_classes:
                        if cls in detected_classes:
                            ep = current_episodes[cls]
                            if not ep:
                                current_episodes[cls] = [(frame_index, frame.copy())]
                            elif frame_index - ep[-1][0] <= 60:
                                ep.append((frame_index, frame.copy()))
                            else:
                                if len(ep) >= 2:
                                    episodes_by_class[cls].append((ep[0], ep[-1]))
                                current_episodes[cls] = [(frame_index, frame.copy())]

                            if max_to_save > 60:
                                output_dir = os.path.join(video_dir, cls)
                                os.makedirs(output_dir, exist_ok=True)
                                timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
                                filename = f"{video_name}_{cls}_{timestamp}_{frame_index}.jpg"
                                filepath = os.path.join(output_dir, filename)
                                cv2.imwrite(filepath, frame)
                                export_counts[cls] += 1

                            detected_by_class[cls].append((frame_index, frame.copy()))

                frame_index += step
                self.progress_bar["value"] = min(frame_index, total_frames)
                self.progress_label.config(text=f"{video_name} Progress: {min(frame_index, total_frames)}/{total_frames}")
                self.root.update_idletasks()

            cap.release()

            for cls in selected_classes:
                if current_episodes[cls] and len(current_episodes[cls]) >= 2:
                    episodes_by_class[cls].append((current_episodes[cls][0], current_episodes[cls][-1]))

                if max_to_save <= 60:
                    all_detected = [ep[0] for ep in episodes_by_class[cls]] + [ep[1] for ep in episodes_by_class[cls]]
                    if len(all_detected) > max_to_save:
                        interval = len(all_detected) / max_to_save
                        indices = [int(i * interval) for i in range(max_to_save)]
                        selected_frames = [all_detected[i] for i in indices]
                    else:
                        selected_frames = all_detected

                    output_dir = os.path.join(video_dir, cls)
                    os.makedirs(output_dir, exist_ok=True)

                    for idx, (f_idx, frame) in enumerate(selected_frames):
                        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
                        filename = f"{video_name}_{cls}_{timestamp}_{f_idx}.jpg"
                        filepath = os.path.join(output_dir, filename)
                        cv2.imwrite(filepath, frame)
                        export_counts[cls] += 1

            # Print export summary for this video
            print(f"Export summary for {video_name}:")
            for cls in selected_classes:
                print(f" - {cls}: {export_counts[cls]} images exported")

        messagebox.showinfo("Done", f"Batch processing complete. Processed {len(self.video_paths)} video(s).")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFrameExtractorApp(root)
    root.mainloop()