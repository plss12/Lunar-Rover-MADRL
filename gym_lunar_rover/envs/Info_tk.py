import tkinter as tk
from tkinter import ttk

class RoverInfoWindow(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.rovers = env.unwrapped.rovers

        self.title("Rover Information")
        self.setup_ui()
        self.refresh_interval = 100
        self.update_info()

    def setup_ui(self):
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Crear el canvas y el scrollbar
        self.canvas = tk.Canvas(main_frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.second_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        self.second_frame.bind("<Configure>", self.on_frame_configure)

        self.rover_frames = []
        for _ in self.rovers:
            rover_frame = ttk.Frame(self.second_frame, padding="10")
            rover_frame.pack(fill=tk.X, pady=5)
            
            name_label = tk.Label(rover_frame, font=("Helvetica", 12))
            name_label.grid(row=0, column=0, sticky="w")
            
            reward_label = tk.Label(rover_frame, font=("Helvetica", 12))
            reward_label.grid(row=1, column=0, sticky="w")
            
            position_label = tk.Label(rover_frame, font=("Helvetica", 12))
            
            mined_label = tk.Label(rover_frame, font=("Helvetica", 12))
            mined_label.grid(row=2, column=0, sticky="w")

            done_label = tk.Label(rover_frame, font=("Helvetica", 12))
            done_label.grid(row=3, column=0, sticky="w")
            
            self.rover_frames.append({
                "frame": rover_frame,
                "name": name_label,
                "reward": reward_label,
                "position": position_label,
                "done": done_label,
                "mined": mined_label
            })

    def on_frame_configure(self, event):
        # Actualiza la región de desplazamiento del canvas para que abarque el marco
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_info(self):
        for i, rover in enumerate(self.rovers):
            rover_frame = self.rover_frames[i]
            rover_frame["name"].config(text=f"Rover {i+1} {rover.position}:")
            rover_frame["reward"].config(text=f"  Reward: {rover.reward}")
            rover_frame["mined"].config(text=f"  Mined: {rover.mined}")
            rover_frame["done"].config(text=f"  Done: {rover.done}")

        self.after(self.refresh_interval, self.update_info)