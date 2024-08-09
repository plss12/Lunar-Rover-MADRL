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

        # Añadir label para la recompensa total del entorno
        self.total_reward_label = tk.Label(main_frame, font=("Helvetica", 14, "bold"))
        self.total_reward_label.pack(side=tk.TOP, pady=10)

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
            
            position_label = tk.Label(rover_frame, font=("Helvetica", 12))

            name_label = tk.Label(rover_frame, font=("Helvetica", 12))
            name_label.grid(row=0, column=0, sticky="w")
            
            total_reward_label = tk.Label(rover_frame, font=("Helvetica", 12))
            total_reward_label.grid(row=1, column=0, sticky="w")
            
            last_reward_label = tk.Label(rover_frame, font=("Helvetica", 12))
            last_reward_label.grid(row=1, column=1, sticky="w")

            mine_pos_label = tk.Label(rover_frame, font=("Helvetica", 12))
            mine_pos_label.grid(row=2, column=0, sticky="w")

            blender_pos_label = tk.Label(rover_frame, font=("Helvetica", 12))
            blender_pos_label.grid(row=2, column=1, sticky="w")
            
            mined_label = tk.Label(rover_frame, font=("Helvetica", 12))
            mined_label.grid(row=3, column=0, sticky="w")

            done_label = tk.Label(rover_frame, font=("Helvetica", 12))
            done_label.grid(row=3, column=1, sticky="w")
            
            self.rover_frames.append({
                "frame": rover_frame,
                "name": name_label,
                "total_reward": total_reward_label,
                "last_reward": last_reward_label,
                "mine_pos":mine_pos_label,
                "blender_pos":blender_pos_label,
                "position": position_label,
                "done": done_label,
                "mined": mined_label
            })

    def on_frame_configure(self, event):
        # Actualiza la región de desplazamiento del canvas para que abarque el marco
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def update_info(self):

        total_reward = self.env.unwrapped.total_reward
        self.total_reward_label.config(text=f"Total Environment Reward: {total_reward}")

        for i, rover in enumerate(self.rovers):
            # Normalizamos la posición a coordenadas x,y reales
            pos = (rover.position[1], self.env.unwrapped.grid_size - rover.position[0] - 1)

            rover_frame = self.rover_frames[i]
            rover_frame["name"].config(text=f"Rover {i+1} {pos}:")
            rover_frame["total_reward"].config(text=f"  Total Reward: {rover.total_reward}")
            rover_frame["last_reward"].config(text=f"  Last Reward: {rover.last_reward}")
            rover_frame["mine_pos"].config(text=f"  Mine Position: {rover.mine_pos}")
            rover_frame["blender_pos"].config(text=f"  Blender Position: {rover.blender_pos}")
            rover_frame["mined"].config(text=f"  Mined: {rover.mined}")
            rover_frame["done"].config(text=f"  Done: {rover.done}")

        self.after(self.refresh_interval, self.update_info)