# This script provides a GUI for rearranging a YOLOv8 dataset into train, validation, and test sets.
# It allows users to select a dataset folder, specify the test set percentage, and adjust the train/validation split.
# The rearrangement is done based on the specified percentages, and a progress bar indicates the operation's progress.
# The script also includes functionality to reset the inputs and clear the progress bar.
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml
import random

class YoloDatasetRearranger:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Dataset Rearranger")

        self.dataset_path = ""
        self.yaml_data = {}

        # Select Dataset Folder
        self.btn_select = tk.Button(root, text="Select Dataset Folder", command=self.select_dataset_folder)
        self.btn_select.pack(pady=10)

        # Test Set Percentage
        self.test_percent_label = tk.Label(root, text="Test Set Percentage:")
        self.test_percent_label.pack()
        self.test_percent_entry = tk.Entry(root)
        self.test_percent_entry.insert(0, "10")
        self.test_percent_entry.pack(pady=5)

        # Trackbar for Train/Validation Split
        self.split_label = tk.Label(root, text="Train/Validation Split:")
        self.split_label.pack()
        self.split_scale = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_split_label)
        self.split_scale.set(70)
        self.split_scale.pack()

        # Dynamic Label for Split Values
        self.dynamic_label = tk.Label(root, text="Train: 63%, Valid: 27%")
        self.dynamic_label.pack(pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        # Rearrange and Reset Buttons
        self.btn_rearrange = tk.Button(root, text="Rearrange", command=self.rearrange_dataset)
        self.btn_rearrange.pack(side=tk.LEFT, padx=10, pady=10)

        self.btn_reset = tk.Button(root, text="Reset", command=self.reset_inputs)
        self.btn_reset.pack(side=tk.RIGHT, padx=10, pady=10)

    def select_dataset_folder(self):
        self.dataset_path = filedialog.askdirectory(title="Select YOLO Dataset Folder")
        if self.dataset_path:
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as file:
                    self.yaml_data = yaml.safe_load(file)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            else:
                messagebox.showerror("Error", "data.yaml not found in the selected folder.")
                self.dataset_path = ""

    def update_split_label(self, val):
        test_percent = int(self.test_percent_entry.get())
        remaining_percent = 100 - test_percent
        train_percent = (self.split_scale.get() / 100) * remaining_percent
        valid_percent = remaining_percent - train_percent
        self.dynamic_label.config(text=f"Train: {train_percent:.1f}%, Valid: {valid_percent:.1f}%")

    def rearrange_dataset(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select a dataset folder.")
            return

        try:
            test_percent = int(self.test_percent_entry.get())
            if not (0 <= test_percent <= 100):
                raise ValueError("Test set percentage must be between 0 and 100.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        remaining_percent = 100 - test_percent
        train_percent = (self.split_scale.get() / 100) * remaining_percent
        valid_percent = remaining_percent - train_percent

        splits = ['train', 'val', 'test']
        image_files = {split: [] for split in splits}
        for split in splits:
            img_folder = os.path.join(self.dataset_path, split, 'images')
            if os.path.exists(img_folder):
                for img_file in os.listdir(img_folder):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        image_files[split].append(os.path.join(img_folder, img_file))

        all_images = image_files['train'] + image_files['val'] + image_files['test']
        random.shuffle(all_images)

        test_count = int(test_percent / 100 * len(all_images))
        test_images = all_images[:test_count]
        remaining_images = all_images[test_count:]

        train_count = int(train_percent / 100 * len(remaining_images))
        train_images = remaining_images[:train_count]
        valid_images = remaining_images[train_count:]

        self.progress['value'] = 0
        self.progress['maximum'] = len(test_images) + len(train_images) + len(valid_images)

        self.move_files('test', test_images)
        self.move_files('train', train_images)
        self.move_files('val', valid_images)

        messagebox.showinfo("Success", f"Dataset rearranged!\nTrain: {len(train_images)} images\nValid: {len(valid_images)} images\nTest: {len(test_images)} images")

    def move_files(self, split, files):
        img_folder = os.path.join(self.dataset_path, split, 'images')
        lbl_folder = os.path.join(self.dataset_path, split, 'labels')
        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(lbl_folder, exist_ok=True)
        for file in files:
            img_filename = os.path.basename(file)
            lbl_filename = os.path.splitext(img_filename)[0] + '.txt'
            shutil.move(file, os.path.join(img_folder, img_filename))
            lbl_file = file.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(lbl_file):
                shutil.move(lbl_file, os.path.join(lbl_folder, lbl_filename))
            self.progress['value'] += 1
            self.root.update_idletasks()

    def reset_inputs(self):
        self.test_percent_entry.delete(0, tk.END)
        self.test_percent_entry.insert(0, "10")
        self.split_scale.set(70)
        self.update_split_label(None)
        self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDatasetRearranger(root)
    root.mainloop()
