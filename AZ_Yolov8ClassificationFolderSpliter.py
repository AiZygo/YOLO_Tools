# This script provides a GUI for splitting a dataset into training, validation, and test sets.
# It allows users to select an input folder containing class subfolders and an output folder for the split datasets.
# Users can specify the percentage of data for each split, and the script will create the necessary
import os
import shutil
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def select_input_folder():
    folder = filedialog.askdirectory(title="Select INPUT folder")
    if folder:
        input_var.set(folder)

def select_output_folder():
    folder = filedialog.askdirectory(title="Select OUTPUT folder")
    if folder:
        output_var.set(folder)

def split_dataset():
    input_dir = input_var.get()
    output_dir = output_var.get()

    if not input_dir or not output_dir:
        messagebox.showerror("Error", "Please select both input and output folders.")
        return

    try:
        train_pct = float(train_var.get()) / 100
        val_pct = float(val_var.get()) / 100
        test_pct = float(test_var.get()) / 100
    except:
        messagebox.showerror("Error", "Invalid percentage values.")
        return

    if abs(train_pct + val_pct + test_pct - 1.0) > 0.01:
        messagebox.showerror("Error", "Train/Val/Test percentages must sum to 100%.")
        return

    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    all_files = []
    for cls in classes:
        cls_path = os.path.join(input_dir, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        for f in files:
            all_files.append((cls, f))

    total_files = len(all_files)
    progress_bar["maximum"] = total_files
    progress_bar["value"] = 0
    root.update_idletasks()

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    for cls in classes:
        files = [f for f in os.listdir(os.path.join(input_dir, cls)) if os.path.isfile(os.path.join(input_dir, cls, f))]
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_pct)
        n_val = int(n_total * val_pct)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        for f in train_files:
            src = os.path.join(input_dir, cls, f)
            dst = os.path.join(output_dir, 'train', cls, f)
            shutil.copy(src, dst)
            progress_bar["value"] += 1
            root.update_idletasks()

        for f in val_files:
            src = os.path.join(input_dir, cls, f)
            dst = os.path.join(output_dir, 'val', cls, f)
            shutil.copy(src, dst)
            progress_bar["value"] += 1
            root.update_idletasks()

        for f in test_files:
            src = os.path.join(input_dir, cls, f)
            dst = os.path.join(output_dir, 'test', cls, f)
            shutil.copy(src, dst)
            progress_bar["value"] += 1
            root.update_idletasks()

    messagebox.showinfo("Done", "Dataset split completed!")

# GUI setup
root = tk.Tk()
root.title("YOLOv8 Classification Dataset Splitter")

input_var = tk.StringVar()
output_var = tk.StringVar()
train_var = tk.StringVar(value="70")
val_var = tk.StringVar(value="20")
test_var = tk.StringVar(value="10")

tk.Button(root, text="Select INPUT Folder", command=select_input_folder).pack(pady=5)
tk.Label(root, textvariable=input_var, wraplength=400).pack()

tk.Button(root, text="Select OUTPUT Folder", command=select_output_folder).pack(pady=5)
tk.Label(root, textvariable=output_var, wraplength=400).pack()

tk.Label(root, text="Train %").pack()
tk.Entry(root, textvariable=train_var).pack()

tk.Label(root, text="Val %").pack()
tk.Entry(root, textvariable=val_var).pack()

tk.Label(root, text="Test %").pack()
tk.Entry(root, textvariable=test_var).pack()

progress_bar = ttk.Progressbar(root, length=400)
progress_bar.pack(pady=10)

tk.Button(root, text="Split Dataset", command=split_dataset).pack(pady=10)

root.mainloop()
