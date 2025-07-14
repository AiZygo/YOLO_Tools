# This script provides a GUI for removing a class from a YOLOv8 classification dataset.
# It allows users to select a data.yaml file, choose a class to remove, and updates the dataset accordingly.
import tkinter as tk
import os
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

def get_class_index(classes, class_name):
    if class_name in classes:
        return classes.index(class_name)
    else:
        return -1

def update_labels(label_dir, old_class_index, class_map, progress_bar, total_files):
    processed_files = 0
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                class_idx = int(parts[0])
                if class_idx != old_class_index:
                    parts[0] = str(class_map[class_idx])
                    new_lines.append(" ".join(parts) + "\n")

            with open(file_path, 'w') as file:
                file.writelines(new_lines)

            processed_files += 1
            progress_bar['value'] = (processed_files / total_files) * 100
            root.update_idletasks()

def select_yaml_file():
    file_path = filedialog.askopenfilename(title="Select data.yaml file", filetypes=[("YAML files", "*.yaml")])
    yaml_path_entry.delete(0, tk.END)
    yaml_path_entry.insert(0, file_path)
    load_classes()

def load_classes():
    yaml_path = yaml_path_entry.get()
    if not yaml_path:
        messagebox.showerror("Error", "Please select a data.yaml file.")
        return

    data = load_yaml(yaml_path)
    classes = data.get('names', [])
    
    class_listbox.delete(0, tk.END)
    for class_name in classes:
        class_listbox.insert(tk.END, class_name)

def process_dataset():
    yaml_path = yaml_path_entry.get()
    if not yaml_path:
        messagebox.showerror("Error", "Please select a data.yaml file.")
        return

    selected_class = class_listbox.get(tk.ACTIVE)
    if not selected_class:
        messagebox.showerror("Error", "Please select a class to remove.")
        return

    # Derive dataset path from the yaml path
    dataset_path = os.path.dirname(yaml_path)

    data = load_yaml(yaml_path)
    classes = data.get('names', [])
    old_class_index = get_class_index(classes, selected_class)

    if old_class_index == -1:
        messagebox.showerror("Error", f"Class '{selected_class}' not found in the dataset.")
        return

    # Remove the class from the classes list and update data.yaml
    classes.pop(old_class_index)
    data['names'] = classes
    data['nc'] = len(classes)  # Update the 'nc' field to the new number of classes
    save_yaml(yaml_path, data)

    # Create a mapping from old class indices to new ones
    class_map = {i: (i if i < old_class_index else i - 1) for i in range(len(classes) + 1)}

    total_files = 0

    # Calculate the total number of label files for the progress bar
    for split in ['train', 'val', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(label_dir):
            total_files += len([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    # Process label files and update them
    for split in ['train', 'val', 'test']:
        label_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(label_dir):
            update_labels(label_dir, old_class_index, class_map, progress_bar, total_files)

    messagebox.showinfo("Success", f"Class '{selected_class}' removed successfully from the dataset.")

# Create the main window
root = tk.Tk()
root.title("YOLOv8 Class Remover")

# YAML file selection
tk.Label(root, text="Select data.yaml file:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
yaml_path_entry = tk.Entry(root, width=50)
yaml_path_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=select_yaml_file).grid(row=0, column=2, padx=10, pady=5)

# Class listbox
tk.Label(root, text="Select class to remove:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
class_listbox = tk.Listbox(root, width=50, height=10)
class_listbox.grid(row=1, column=1, padx=10, pady=5, rowspan=4)

# Process button
tk.Button(root, text="Remove Class", command=process_dataset).grid(row=5, column=1, padx=10, pady=20)

# Progress bar
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=6, column=0, columnspan=3, padx=10, pady=20)

# Run the Tkinter event loop
root.mainloop()
