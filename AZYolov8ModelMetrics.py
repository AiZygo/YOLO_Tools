# exports some model metrics and the jpg files created by the yolov8
# and save the results in a JSON file.
# This script provides a GUI for visualizing YOLOv8 model metrics.
# It allows users to select a model and dataset, validate the model, visualize metrics, and save/load results.
# The visualization includes F1, Precision, Recall, and mAP curves.

import json
import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO


class YoloModelVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Model Metrics Visualizer")

        self.model_path = ""
        self.data_path = ""
        self.results = None
        self.encodedresults=[]

        # Select Model Button
        self.btn_select_model = tk.Button(root, text="1.Select .pt Model", command=self.select_model)
        self.btn_select_model.pack(pady=10)

        # Select Data Button
        self.btn_select_data = tk.Button(root, text="2.Select data.yaml", command=self.select_data)
        self.btn_select_data.pack(pady=10)
        
        # Validate Button
        self.btn_visualize = tk.Button(root, text="3.Validate Metrics", command=self.val_metrics)
        self.btn_visualize.pack(pady=10)

        # Save Results Button
        self.btn_save = tk.Button(root, text="Save Results", command=self.save_results)
        self.btn_save.pack(pady=10)

        # Load Results Button
        self.btn_load = tk.Button(root, text="Load Results", command=self.load_results)
        self.btn_load.pack(pady=10)

        # Visualization Frame
        self.visualization_frame = tk.Frame(root)
        self.visualization_frame.pack(fill='both', expand=True)

        # Visualize Button
        self.btn_visualize = tk.Button(root, text="4.Visualize Metrics", command=self.visualize_metrics)
        self.btn_visualize.pack(pady=10)

    def encode(self):
        # List all attributes of ap.b
        attributes = dir(self.results.box)

        # Filter attributes that you are interested in
        # In this example, we are interested in attributes that do not start with '__'
        filtered_attributes = [attr for attr in attributes if not attr.startswith('__')]

        # Iterate through the filtered attributes and print their values
        self.encodedresults={}

        for attr in filtered_attributes:
            value = getattr(self.results.box, attr)
            if isinstance(value,np.ndarray):
                
                valuelist=value.tolist()
            self.encodedresults[attr]=valuelist

    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select YOLOv8 Model File", filetypes=[("PyTorch files", "*.pt")])
        if self.model_path:
            messagebox.showinfo("Success", "Model file selected successfully!")
        else:
            messagebox.showerror("Error", "No model file selected.")

    def select_data(self):
        self.data_path = filedialog.askopenfilename(title="Select Dataset Configuration File", filetypes=[("YAML files", "*.yaml")])
        if self.data_path:
            messagebox.showinfo("Success", "Data configuration file selected successfully!")
        else:
            messagebox.showerror("Error", "No data configuration file selected.")

    def val_metrics(self):
        if not self.model_path or not os.path.exists(self.model_path):
            messagebox.showerror("Error", "Invalid model file.")
            return

        if not self.data_path or not os.path.exists(self.data_path):
            messagebox.showerror("Error", "Invalid data configuration file.")
            return

        try:
            model = YOLO(self.model_path)
            self.results = model.val(data=self.data_path,save_json=True)  # Run the evaluation
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
        
        self.encode()
    
    def visualize_metrics(self):
        if not self.results:
            messagebox.showerror("Error", "No metrics to visualize. Please load or validate metrics first.")
            return

        self.clear_visualization()

        # Example of visualizing some metrics
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        if self.results['f1']:
            for c in range(len(self.results['f1_curve'])):
                axs[0, 0].plot(self.results['f1_curve'][c])
                axs[0, 0].set_title('F1')
        
        if self.results['p']:
            for c in range(len(self.results['p_curve'])):
                axs[0, 1].plot(self.results['p_curve'][c])
                axs[0, 1].set_title('Precision')
                
        if self.results['r']:
            for c in range(len(self.results['r_curve'])):
                axs[1, 0].plot(self.results['r_curve'][c])
                axs[1, 0].set_title('Recall')

        if self.results['map75']:
            for c in range(len(self.results['map75'])):
                axs[1, 1].plot(self.results['map75'][c])
                axs[1, 1].set_title('map75')

        for ax in axs.flat:
            ax.set(xlabel='Epoch', ylabel='Value')

        for ax in axs.flat:
            ax.label_outer()

        canvas = FigureCanvasTkAgg(fig, master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_visualization(self):
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

    def save_results(self):
        if not self.results:
            messagebox.showerror("Error", "No results to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if save_path:
            with open(save_path, 'w') as f:
                self.encode()
                json.dump(self.encodedresults, f)  # Save results as dictionary with NumpyEncoder
            messagebox.showinfo("Success", "Results saved successfully!")

    def load_results(self):
        load_path = filedialog.askopenfilename(title="Load Results File", filetypes=[("JSON files", "*.json")])
        if load_path:
            with open(load_path, 'r') as f:
                self.results = json.load(f)  # Load results with numpy_decoder
            messagebox.showinfo("Success", "Results loaded successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = YoloModelVisualizer(root)
    root.mainloop()
