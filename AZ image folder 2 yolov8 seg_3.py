# This script provides a GUI for selecting an image folder and a YOLOv8 model file.
# It processes images in the selected folder, generates segmentation labels as samenamefile.txt, 
# and saves them in a label folder. Creates a data.yaml file in the root folder.
# it can porcess images recursively in subfolders.
# also has polygon simplification using cv2.approxPolyDP.
# The segmentation results are displayed in a viewer with navigation controls.
# has been modified to include a progress bar and a viewer for the segmentation results.

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class YOLOSegApp:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLOv8 Segmentation GUI")

        self.image_folder = ''
        self.model_path = ''
        self.model = None

        # GUI layout
        tk.Button(master, text="Select Image Folder", command=self.select_image_folder).pack(pady=5)
        tk.Button(master, text="Select YOLOv8 Model (.pt)", command=self.select_model).pack(pady=5)

        self.epsilon_value = tk.DoubleVar(value=0.002)
        tk.Label(master, text="Polygon Simplification (ε)").pack()
        tk.Scale(master, from_=0.001, to=0.01, resolution=0.0005, orient='horizontal',
                 variable=self.epsilon_value, length=300).pack(pady=5)

        tk.Button(master, text="Start Processing", command=self.start_processing).pack(pady=10)
        tk.Button(master, text="View Results", command=self.view_results).pack(pady=5)

        self.progress = ttk.Progressbar(master, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=5)

    def select_image_folder(self):
        self.image_folder = filedialog.askdirectory()
        if self.image_folder:
            messagebox.showinfo("Folder Selected", f"Image folder:\n{self.image_folder}")

    def select_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("YOLOv8 Model", "*.pt")])
        if self.model_path:
            self.model = YOLO(self.model_path)
            messagebox.showinfo("Model Loaded", f"Model loaded:\n{self.model_path}")

    def start_processing(self):
        if not self.image_folder or not self.model:
            messagebox.showerror("Error", "Please select both image folder and model.")
            return

        # Create data.yaml in root
        yaml_path = os.path.join(self.image_folder, "data.yaml")
        class_list = self.model.names.values()
        with open(yaml_path, 'w') as f:
            f.write("path: .\ntrain: images\nval: images\n")
            f.write(f"nc: {len(class_list)}\n")
            f.write("names: [" + ", ".join(f"'{n}'" for n in class_list) + "]\n")

        image_files = []
        all_dirs = list(os.walk(self.image_folder))
        self.progress["maximum"] = len(all_dirs)
        self.progress["value"] = 0
        self.master.update_idletasks()

        for idx, (root_dir, _, files) in enumerate(all_dirs):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_files.append((os.path.join(root_dir, file), root_dir))
            self.progress["value"] = idx + 1
            self.master.update_idletasks()

        self.progress["maximum"] = len(image_files)
        self.progress["value"] = 0

        for idx, (image_path, img_folder) in enumerate(image_files):
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            results = self.model.predict(image, task="segment", save=False, verbose=False)[0]

            label_lines = []
            if results.masks is not None and results.masks.xy:
                for seg, cls in zip(results.masks.xy, results.boxes.cls):
                    contour = seg.reshape((-1, 1, 2)).astype("float32")
                    epsilon = self.epsilon_value.get() * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    normalized = [(x[0][0] / w, x[0][1] / h) for x in approx]
                    flat_coords = [f"{x:.6f} {y:.6f}" for x, y in normalized]
                    label_lines.append(f"{int(cls)} " + " ".join(flat_coords))

            label_dir = os.path.join(img_folder, "labels")
            os.makedirs(label_dir, exist_ok=True)
            txt_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(label_lines))

            self.progress["value"] = idx + 1
            self.master.update_idletasks()

        messagebox.showinfo("Done", f"Processed {len(image_files)} images.\nSegmentation results saved.")

    def view_results(self):
        if not self.image_folder:
            messagebox.showerror("Error", "No folder selected.")
            return

        image_paths = []
        all_dirs = list(os.walk(self.image_folder))
        self.progress["maximum"] = len(all_dirs)
        self.progress["value"] = 0
        self.master.update_idletasks()

        for idx, (root_dir, _, files) in enumerate(all_dirs):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    label_path = os.path.join(root_dir, "labels", os.path.splitext(file)[0] + ".txt")
                    if os.path.exists(label_path):
                        image_paths.append((os.path.join(root_dir, file), label_path))
            self.progress["value"] = idx + 1
            self.master.update_idletasks()

        if not image_paths:
            messagebox.showinfo("No Results", "No labeled images found.")
            return

        viewer = tk.Toplevel(self.master)
        viewer.title("Segmentation Viewer")
        viewer.geometry("1000x750")

        canvas_frame = tk.Frame(viewer)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        state = {"index": 0, "images": image_paths, "canvas": None}

        def show_image():
            viewer_img_path, viewer_lbl_path = state["images"][state["index"]]
            img = cv2.cvtColor(cv2.imread(viewer_img_path), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.set_title(os.path.basename(viewer_img_path))

            with open(viewer_lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    polygon = np.array(coords).reshape(-1, 2)
                    polygon[:, 0] *= w
                    polygon[:, 1] *= h
                    ax.add_patch(patches.Polygon(polygon, closed=True, fill=False, edgecolor='lime', linewidth=1.5))
                    ax.text(polygon[0][0], polygon[0][1], str(cls), color='yellow', fontsize=9)

            ax.axis('off')
            if state["canvas"]:
                state["canvas"].get_tk_widget().destroy()

            state["canvas"] = FigureCanvasTkAgg(fig, master=canvas_frame)
            state["canvas"].draw()
            state["canvas"].get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def next_img(event=None):
            state["index"] = (state["index"] + 1) % len(state["images"])
            show_image()

        def prev_img(event=None):
            state["index"] = (state["index"] - 1) % len(state["images"])
            show_image()

        viewer.bind("<Right>", next_img)
        viewer.bind("<Left>", prev_img)
        show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOSegApp(root)
    root.mainloop()
