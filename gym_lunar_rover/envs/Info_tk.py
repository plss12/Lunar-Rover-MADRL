import tkinter as tk
from tkinter import ttk

class RoverInfoWindow:
    def __init__(self, root, env):
        self.env = env
        self.rovers = env.rovers
        self.root = root
        self.root.title("Rover Information")

        # Crear un marco con scrollbar
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.second_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.second_frame, anchor="nw")

        self.update_info()

    def update_info(self):
        for widget in self.second_frame.winfo_children():
            widget.destroy()

        for i, rover in enumerate(self.rovers):
            tk.Label(self.second_frame, text=f"Rover {i+1}:", font=("Helvetica", 12)).grid(row=i*3, column=0, sticky="w")
            tk.Label(self.second_frame, text=f"  Reward: {rover.reward}", font=("Helvetica", 12)).grid(row=i*3+1, column=0, sticky="w")
            tk.Label(self.second_frame, text=f"  Position: {rover.position}", font=("Helvetica", 12)).grid(row=i*3+2, column=0, sticky="w")

    def refresh(self):
        self.update_info()
        self.root.after(1000, self.refresh)