# This script provides a GUI for analyzing video files using the YOLOv8 model.
# It allows users to load a YOLOv8 model, select a video file, and process the video to detect objects.
# The results include class counts, frames where classes are detected, and bounding box centers.
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import pandas as pd
import datetime
import os
import numpy as np
import json

class YOLOVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Video Analysis")
        self.yolo_model = None
        self.step = 0
        self.counter = 0
        self.frame_skip = 20  # Default value
        self.classes_count = {}
        self.classes_frames = []
        self.class_centers = {}
        self.stop_process = False  # Flag to stop the process
        self.create_widgets()

    def create_widgets(self):
        self.load_model_button = tk.Button(self.root, text="Load YOLOv8 Model", command=self.load_model)
        self.load_model_button.pack()

        self.load_video_button = tk.Button(self.root, text="Load Video", command=self.load_video, state=tk.DISABLED)
        self.load_video_button.pack()

        self.start_button = tk.Button(self.root, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Stop Process", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack()

        self.show_analysis_button = tk.Button(self.root, text="Show Analysis", command=self.show_analysis, state=tk.DISABLED)
        self.show_analysis_button.pack()

        self.skip_label = tk.Label(self.root, text="Read every x frames:")
        self.skip_label.pack()
        self.skip_entry = tk.Entry(self.root)
        self.skip_entry.pack()
        self.skip_entry.insert(0, "20")

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack()

        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PyTorch Model files", "*.pt")])
        if model_path:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to('cuda')
            messagebox.showinfo("Model Loaded", f"Loaded model from {model_path}")
            self.load_video_button.config(state=tk.NORMAL)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if self.video_path:
            self.start_button.config(state=tk.NORMAL)

    def start_processing(self):
        try:
            self.frame_skip = int(self.skip_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for frame skip.")
            return
        self.stop_process = False  # Reset the stop flag
        self.stop_button.config(state=tk.NORMAL)
        self.show_analysis_button.config(state=tk.DISABLED)
        threading.Thread(target=self.process_video).start()

    def stop_processing(self):
        self.stop_process = True
        self.stop_button.config(state=tk.DISABLED)
        self.show_analysis_button.config(state=tk.NORMAL)

    def process_video(self):
        if not self.yolo_model:
            messagebox.showerror("Error", "No YOLOv8 model loaded.")
            return

        cap = cv2.VideoCapture(self.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        self.progress["maximum"] = frame_count // self.frame_skip  # Adjust the progress bar maximum

        processed_frame_number = 0  # To track every nth frame
        names = self.yolo_model.names
        frame_number = 1
        while cap.isOpened():
            if self.stop_process:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            frame_number += self.frame_skip
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))  # Resize the frame immediately after reading

            results = self.yolo_model(frame, iou=0.3, max_det=12, conf=0.25, verbose=False)
            detected_classes = []
            for r in results:
                for box in r.boxes:
                    class_name = names[int(box.cls)]
                    detected_classes.append(class_name)
                    self.classes_count[class_name] = self.classes_count.get(class_name, 0) + 1
                    self.classes_frames.append((frame_number, class_name, frame_number / frame_rate / 60))
                    center_x = round((box.xyxy.tolist()[0][0] + box.xyxy.tolist()[0][2]) / 2)
                    center_y = round((box.xyxy.tolist()[0][1] + box.xyxy.tolist()[0][3]) / 2)
                    if class_name not in self.class_centers:
                        self.class_centers[class_name] = []
                    self.class_centers[class_name].append((center_x, center_y))

            # Draw results on frame
            display_frame = results[0].plot()
            self.root.after(0, self.display_frame, display_frame)

            processed_frame_number += 1
            self.root.after(0, self.update_progress, processed_frame_number, frame_count // self.frame_skip)

        cap.release()
        self.stop_button.config(state=tk.DISABLED)
        self.show_analysis_button.config(state=tk.NORMAL)
        self.root.after(0, self.display_results)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.root.update_idletasks()
        self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

    def update_progress(self, current_frame, total_frames):
        self.progress["value"] = current_frame
        self.progress_label.config(text=f"Processing frame {current_frame}/{total_frames}")
        self.root.update_idletasks()

    def display_results(self):
        pass  # Do nothing here, only show results when show_analysis is called

    def show_analysis(self):
        # Create a new window for the tabs
        results_window = tk.Toplevel(self.root)
        results_window.title("Results")
        results_window.geometry("800x600")
        
        save_button = tk.Button(results_window, text="Save Graphs", command=self.save_graphs)
        save_button.pack(side=tk.BOTTOM)

        notebook = ttk.Notebook(results_window)
        notebook.pack(expand=1, fill='both')

        self.tab1 = ttk.Frame(notebook)
        self.tab2 = ttk.Frame(notebook)
        self.tab3 = ttk.Frame(notebook)
        self.tab4 = ttk.Frame(notebook)

        notebook.add(self.tab1, text='Class Count')
        notebook.add(self.tab2, text='Class vs Frame')
        notebook.add(self.tab3, text='Class vs Time (minutes)')
        notebook.add(self.tab4, text='Bounding Box Centers')

        self.display_class_count(self.tab1)
        self.display_class_vs_frame(self.tab2)
        self.display_class_vs_time(self.tab3)
        self.display_bbox_centers(self.tab4)

    def display_class_count(self, tab):
        fig, ax = plt.subplots()
        class_names = list(self.classes_count.keys())
        counts = list(self.classes_count.values())
        ax.barh(class_names, counts)
        ax.set_xlabel('Count')
        ax.set_title('Class Count')
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        tab.fig = fig  # Keep a reference to the figure

    def display_class_vs_frame(self, tab):
        df = pd.DataFrame(self.classes_frames, columns=['Frame', 'Class', 'Time'])
        fig, ax = plt.subplots()
        for class_name in df['Class'].unique():
            class_df = df[df['Class'] == class_name]
            ax.scatter(class_df['Frame'], class_df['Class'], label=class_name)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Class')
        ax.set_title('Class vs Frame')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        tab.fig = fig  # Keep a reference to the figure

    def display_class_vs_time(self, tab):
        df = pd.DataFrame(self.classes_frames, columns=['Frame', 'Class', 'Time'])
        fig, ax = plt.subplots()
        for class_name in df['Class'].unique():
            class_df = df[df['Class'] == class_name]
            ax.scatter(class_df['Time'], class_df['Class'], label=class_name)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Class')
        ax.set_title('Class vs Time (minutes)')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        tab.fig = fig  # Keep a reference to the figure

    def display_bbox_centers(self, tab):
        class_names = list(self.class_centers.keys())
        selected_class = tk.StringVar(tab)
        selected_class.set(class_names[0])  # Set default value

        def update_heatmap(*args):
            class_name = selected_class.get()
            centers = self.class_centers[class_name]
            if hasattr(update_heatmap, "canvas"):
                update_heatmap.canvas.get_tk_widget().pack_forget()  # Remove the previous plot if it exists

            # Recalculate the centers to adjust for the top-left origin
            adjusted_centers = [(x, 480 - y) for x, y in centers]

            # Create a DataFrame for seaborn
            df = pd.DataFrame(adjusted_centers, columns=['x', 'y'])

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.clear()  # Clear the axes to remove the previous plot

            if len(df) > 1:
                try:
                    sns.kdeplot(data=df, x='x', y='y', cmap='Blues', fill=True, thresh=0, levels=100, ax=ax)
                    ax.set_title(f'Bounding Box Centers for {class_name}')
                    ax.set_xlim(0, 640)
                    ax.set_ylim(0, 480)
                except ValueError as e:
                    ax.text(0.5, 0.5, str(e), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title(f'Bounding Box Centers for {class_name}')
                    ax.set_xlim(0, 640)
                    ax.set_ylim(0, 480)
            else:
                ax.text(0.5, 0.5, 'Not enough data to generate heatmap', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f'Bounding Box Centers for {class_name}')
                ax.set_xlim(0, 640)
                ax.set_ylim(0, 480)

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            tab.fig = fig  # Keep a reference to the figure
            update_heatmap.canvas = canvas  # Store the canvas reference to delete it later

            plt.close(fig)  # Close the figure to prevent too many open figures

        class_menu = ttk.OptionMenu(tab, selected_class, class_names[0], *class_names)
        class_menu.pack(side=tk.TOP, fill=tk.X)
        selected_class.trace("w", update_heatmap)
        update_heatmap()

    def save_graphs(self):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        save_path = filedialog.askdirectory()
        if save_path:
            self.save_figure(self.tab1, os.path.join(save_path, f'classcount_{timestamp}.jpg'))
            self.save_figure(self.tab2, os.path.join(save_path, f'classvsframe_{timestamp}.jpg'))
            self.save_figure(self.tab3, os.path.join(save_path, f'classvstime_{timestamp}.jpg'))
            self.save_figure(self.tab4, os.path.join(save_path, f'bboxcenters_{timestamp}.jpg'))
       
        with open(os.path.join(save_path, f'class_data_{timestamp}.json'), 'w') as f:
            json.dump(self.classes_frames, f)  # Save results as dictionary with NumpyEncoder
        messagebox.showinfo("Success", "Results saved successfully!")

    def save_figure(self, tab, path):
        tab.fig.savefig(path)
        messagebox.showinfo("Save Graph", f"Graph saved to {path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOVideoApp(root)
    root.mainloop()
