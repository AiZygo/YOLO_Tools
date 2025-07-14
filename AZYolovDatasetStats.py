# This script provides a GUI for visualizing YOLOv8 dataset statistics.
# It allows users to select a dataset folder, view class counts, and sort them by index or popularity.
# The statistics are displayed in a tabbed interface for all, train, validation, and test datasets.
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

class YoloDatasetVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Dataset Visualizer")

        self.dataset_path = ""
        self.yaml_data = {}
        self.class_counts = {'all': {}, 'train': {}, 'val': {}, 'test': {}}
        self.sorted_class_counts = []

        # Select Dataset Folder Button
        self.btn_select = tk.Button(root, text="Select Dataset Folder", command=self.select_dataset_folder)
        self.btn_select.pack(pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        # Sorting Buttons
        self.sort_frame = tk.Frame(root)
        self.sort_frame.pack(pady=10)
        self.btn_sort_index = tk.Button(self.sort_frame, text="Sort by Index", command=self.sort_by_index)
        self.btn_sort_index.pack(side=tk.LEFT, padx=5)
        self.btn_sort_popularity = tk.Button(self.sort_frame, text="Sort by Popularity", command=self.sort_by_popularity)
        self.btn_sort_popularity.pack(side=tk.LEFT, padx=5)

        # Notebook for Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        # Tabs
        self.tab_all = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_val = ttk.Frame(self.notebook)
        self.tab_test = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_all, text="All")
        self.notebook.add(self.tab_train, text="Train")
        self.notebook.add(self.tab_val, text="Val")
        self.notebook.add(self.tab_test, text="Test")

    def select_dataset_folder(self):
        self.dataset_path = filedialog.askdirectory(title="Select YOLO Dataset Folder")
        if self.dataset_path:
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as file:
                    self.yaml_data = yaml.safe_load(file)
                self.load_class_counts()
                self.sort_by_index()
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            else:
                messagebox.showerror("Error", "data.yaml not found in the selected folder.")
                self.dataset_path = ""

    def load_class_counts(self):
        self.class_counts = {'all': {}, 'train': {}, 'val': {}, 'test': {}}
        class_names = self.yaml_data.get('names', [])

        total_files = 0
        for split in ['train', 'val', 'test']:
            label_folder = os.path.join(self.dataset_path, split, 'labels')
            if os.path.exists(label_folder):
                total_files += len([f for f in os.listdir(label_folder) if f.endswith('.txt')])

        processed_files = 0
        self.progress['value'] = 0
        self.progress['maximum'] = total_files

        for split in ['train', 'val', 'test']:
            img_folder = os.path.join(self.dataset_path, split, 'images')
            label_folder = os.path.join(self.dataset_path, split, 'labels')

            if os.path.exists(label_folder):
                for label_file in os.listdir(label_folder):
                    if label_file.endswith('.txt'):
                        with open(os.path.join(label_folder, label_file), 'r') as file:
                            for line in file:
                                class_id = int(line.strip().split()[0])
                                class_name = class_names[class_id]
                                if class_name not in self.class_counts[split]:
                                    self.class_counts[split][class_name] = 0
                                self.class_counts[split][class_name] += 1

                                if class_name not in self.class_counts['all']:
                                    self.class_counts['all'][class_name] = 0
                                self.class_counts['all'][class_name] += 1

                    processed_files += 1
                    self.progress['value'] = processed_files
                    self.root.update_idletasks()

    def sort_by_index(self):
        class_names = self.yaml_data.get('names', [])
        self.sorted_class_counts = sorted(self.class_counts['all'].items(), key=lambda item: class_names.index(item[0]))
        self.display_class_counts()

    def sort_by_popularity(self):
        self.sorted_class_counts = sorted(self.class_counts['all'].items(), key=lambda item: item[1], reverse=True)
        self.display_class_counts()

    def display_class_counts(self):
        for tab, split in [(self.tab_all, 'all'), (self.tab_train, 'train'), (self.tab_val, 'val'), (self.tab_test, 'test')]:
            for widget in tab.winfo_children():
                widget.destroy()

            class_counts = self.class_counts[split]
            if split == 'all':
                sorted_counts = self.sorted_class_counts
            else:
                sorted_counts = sorted(class_counts.items(), key=lambda item: self.sorted_class_counts.index((item[0], self.class_counts['all'][item[0]])))

            row = 0
            for class_name, count in sorted_counts:
                tk.Label(tab, text=f"{class_name}: {count}").grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
                row += 1

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDatasetVisualizer(root)
    root.mainloop()
