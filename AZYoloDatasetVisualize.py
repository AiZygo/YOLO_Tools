# Visualize annotations of BBoxes or Polygons 
# in a YOLO dataset using a GUI application.
# The script allows users to view images and their annotations, switch between bounding boxes and polygons,
import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import yaml

class YoloDatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Dataset Viewer")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        self.btn_prev = tk.Button(self.btn_frame, text="Previous", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT)

        self.btn_next = tk.Button(self.btn_frame, text="Next", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT)

        self.btn_select = tk.Button(self.btn_frame, text="Select Dataset Folder", command=self.select_dataset_folder)
        self.btn_select.pack(side=tk.LEFT)

        self.annotation_type_var = tk.StringVar(value="auto")
        self.radio_bbox = tk.Radiobutton(self.btn_frame, text="Bounding Boxes", variable=self.annotation_type_var, value="bbox")
        self.radio_bbox.pack(side=tk.LEFT)

        self.radio_polygon = tk.Radiobutton(self.btn_frame, text="Polygons", variable=self.annotation_type_var, value="polygon")
        self.radio_polygon.pack(side=tk.LEFT)

        self.radio_auto = tk.Radiobutton(self.btn_frame, text="Auto Detect", variable=self.annotation_type_var, value="auto")
        self.radio_auto.pack(side=tk.LEFT)

        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        self.image_list = []
        self.annotation_list = []
        self.class_names = []
        self.class_colors = {}
        self.current_index = 0

    def select_dataset_folder(self):
        folder_path = filedialog.askdirectory(title="Select YOLO Dataset Folder")
        if folder_path:
            yaml_path = os.path.join(folder_path, 'data.yaml')
            if os.path.exists(yaml_path):
                self.load_classes(yaml_path)
                self.load_dataset(folder_path)
                self.show_image()
            else:
                messagebox.showerror("Error", "data.yaml not found in the selected folder.")

    def load_classes(self, yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
            self.class_names = data.get('names', [])
            self.class_colors = {i: "#{:06x}".format(random.randint(0, 0xFFFFFF)) for i in range(len(self.class_names))}

    def load_dataset(self, folder_path):
        self.image_list = []
        self.annotation_list = []

        for split in ['train', 'val', 'test']:
            img_folder = os.path.join(folder_path, split, 'images')
            label_folder = os.path.join(folder_path, split, 'labels')

            if os.path.exists(img_folder) and os.path.exists(label_folder):
                for img_file in os.listdir(img_folder):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        self.image_list.append(os.path.join(img_folder, img_file))
                        label_file = os.path.join(label_folder, os.path.splitext(img_file)[0] + '.txt')
                        if os.path.exists(label_file):
                            self.annotation_list.append(self.load_annotations(label_file))
                        else:
                            self.annotation_list.append([])

    def load_annotations(self, label_file):
        annotations = []
        annotation_type = self.annotation_type_var.get()
        
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                class_id = int(parts[0])
                points = list(map(float, parts[1:]))

                if annotation_type == "auto":
                    if len(points) == 4:
                        annotation_type = "bbox"
                    else:
                        annotation_type = "polygon"

                if annotation_type == "bbox":
                    bbox = points
                    annotations.append((class_id, bbox))
                elif annotation_type == "polygon":
                    polygon = points
                    annotations.append((class_id, polygon))
                    
        return annotations

    def show_image(self):
        if not self.image_list:
            return

        img_path = self.image_list[self.current_index]
        img = Image.open(img_path)
        img = img.resize((800, 600), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)

        self.canvas.image = img_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        self.draw_annotations()

    def draw_annotations(self):
        annotations = self.annotation_list[self.current_index]
        annotation_type = self.annotation_type_var.get()

        for class_id, points in annotations:
            class_name = self.class_names[class_id]
            color = self.class_colors[class_id]
            if annotation_type == "bbox" or (annotation_type == "auto" and len(points) == 4):
                x_center, y_center, width, height = points
                x_center *= 800
                y_center *= 600
                width *= 800
                height *= 600

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                self.canvas.create_text(x_center, y_center, text=class_name, fill=color, font=("Helvetica", 12), anchor=tk.CENTER)
            elif annotation_type == "polygon" or (annotation_type == "auto" and len(points) > 4):
                scaled_points = [p * 800 if i % 2 == 0 else p * 600 for i, p in enumerate(points)]
                self.canvas.create_polygon(scaled_points, outline=color, width=2, fill='')

                # Calculate the barycenter
                x_coords = scaled_points[0::2]
                y_coords = scaled_points[1::2]
                barycenter_x = sum(x_coords) / len(x_coords)
                barycenter_y = sum(y_coords) / len(y_coords)

                self.canvas.create_text(barycenter_x, barycenter_y, text=class_name, fill=color, font=("Helvetica", 12), anchor=tk.CENTER)

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.show_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloDatasetViewer(root)
    root.mainloop()
