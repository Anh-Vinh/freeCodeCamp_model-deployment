import mediapipe as mp
import tkinter as tk
import landmarkers
import json
import cv2
import os

from tkinter import ttk, filedialog, messagebox

class ConvertApp():
    def __init__(self, root):
        self.root = root
        self.root.title("Video to JSON Converter")
        self.root.geometry("1400x600")
        self.root.grid_columnconfigure(0, weight=1)  # Left panel (video)
        self.root.grid_columnconfigure(1, weight=0)  # Middle (progress)
        self.root.grid_columnconfigure(2, weight=1)  # Right panel (JSON)
        self.root.grid_rowconfigure(2, weight=1)     # Make list areas expand
        self.face_landmarker = None
        self.hands_landmarker = None
        self.pose_landmarker= None

        # Variables
        self.video_folder = tk.StringVar()
        self.json_folder = tk.StringVar()

        # ===== Left: Video folder =====
        video_frame = tk.Frame(root, padx=10, pady=10)
        video_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")

        tk.Label(video_frame, text="Video Folder:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Entry(video_frame, textvariable=self.video_folder, width=60).pack(fill="x", pady=5)
        tk.Button(video_frame, text="Browse", command=self.browse_video).pack(pady=(0, 10))

        self.video_list = tk.Listbox(video_frame, height=20)
        self.video_scroll = tk.Scrollbar(video_frame, orient="vertical", command=self.video_list.yview)
        self.video_list.config(yscrollcommand=self.video_scroll.set)
        self.video_list.pack(side="left", fill="both", expand=True)
        self.video_scroll.pack(side="right", fill="y")

        # ===== Middle: Progress bar =====
        progress_frame = tk.Frame(root, padx=20, pady=150)
        progress_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")
        progress_frame.grid_rowconfigure(1, weight=1)

        self.face = tk.BooleanVar()
        self.hands = tk.BooleanVar()
        self.pose = tk.BooleanVar()

        tk.Checkbutton(progress_frame, text="Face", variable=self.face).pack(anchor="w", padx=100, pady=5)
        tk.Checkbutton(progress_frame, text="Hands", variable=self.hands).pack(anchor="w", padx=100, pady=5)
        tk.Checkbutton(progress_frame, text="Pose", variable=self.pose).pack(anchor="w", padx=100, pady=5)


        self.progress = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate")
        self.progress.pack(expand=True)

        self.convert_button = tk.Button(root, text="Convert", command=self.process_folder, width=15, bg="#4CAF50", fg="white")
        self.convert_button.grid(row=2, column=1, pady=10)

        # ===== Right: JSON folder =====
        json_frame = tk.Frame(root, padx=10, pady=10)
        json_frame.grid(row=0, column=2, rowspan=3, sticky="nsew")

        tk.Label(json_frame, text="JSON Folder:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Entry(json_frame, textvariable=self.json_folder, width=60).pack(fill="x", pady=5)
        tk.Button(json_frame, text="Browse", command=self.browse_json).pack(pady=(0, 10))

        self.json_list = tk.Listbox(json_frame, height=20)
        self.json_scroll = tk.Scrollbar(json_frame, orient="vertical", command=self.json_list.yview)
        self.json_list.config(yscrollcommand=self.json_scroll.set)
        self.json_list.pack(side="left", fill="both", expand=True)
        self.json_scroll.pack(side="right", fill="y")
    
    def update_file_list(self, folder, listbox, extensions):
        listbox.delete(0, tk.END)
        files = [f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in extensions)]
        for file in sorted(files):
            listbox.insert(tk.END, file)

    def browse_video(self):
        folder = filedialog.askdirectory(title="Select video folder")
        if folder:
            self.video_folder.set(folder)
            self.update_file_list(folder, self.video_list, [".mp4", ".avi", ".mov"])

    def browse_json(self):
        folder = filedialog.askdirectory(title="Select JSON folder")
        if folder:
            self.json_folder.set(folder)
            self.update_file_list(folder, self.json_list, [".json"])

    def create_task(self):
        if self.face.get():
            self.face_landmarker.create_task()

        if self.hands.get():
            self.hands_landmarker.create_task()

        if self.pose.get():
            self.pose_landmarker.create_task()

    def close_task(self):
        [lm.close_task() for lm in [self.face_landmarker, self.hands_landmarker, self.pose_landmarker] if lm]

    
    def process_video(self, video_path):
        landmarks_data_list = []

        # Load video
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_index = 0

        # Loop through the video
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_timestamp_ms = int((frame_index / fps) * 1000)

            # Detect landmarks in the frame
            if self.face:
                face_landmark = self.face_landmarker.get_empty_face()
            else:
                face_landmark = self.face_landmarker.process_frame(mp_frame, frame_timestamp_ms)

            if self.hands:
                hand_landmark = self.hands_landmarker.get_empty_hands()
            else:
                hand_landmark = self.hand_landmarker.process_frame(mp_frame, frame_timestamp_ms)

            if self.pose:
                pose_landmark = self.pose_landmarker.get_empty_pose()
            else:
                pose_landmark = self.pose_landmarker.process_frame(mp_frame, frame_timestamp_ms)

            # Update the landmark_data_list each frame
            landmarks_data_list.extend(face_landmark)
            landmarks_data_list.extend(hand_landmark)
            landmarks_data_list.extend(pose_landmark)

            frame_index += 1

        video.release()
        return landmarks_data_list

    
    def process_folder(self):
        
        # Create tasks accordingly
        if self.video_folder and self.json_folder:
            # Create landmarkers
            self.face_landmarker = landmarkers.FaceLandmarker()
            self.hands_landmarker = landmarkers.HandsLandmarker()
            self.pose_landmarker = landmarkers.PoseLandmarker()

            self.create_task()
            
            video_list = os.listdir(self.video_folder.get())
            for video_file in video_list:
                video_path = os.path.join(self.video_folder.get(), video_file)
                landmarks_data_list = self.process_video(video_path)
                
                # Save as json format
                json_path = os.path.join(self.json_folder.get(), video_file.split(".")[0] + ".json")
                with open(json_path, "w") as json_file:
                    data = {'landmarks_data_list': landmarks_data_list}
                    json.dump(data, json_file, indent=2)

            self.close_task()

if __name__ == '__main__':
    root = tk.Tk()
    app = ConvertApp(root)
    root.mainloop()