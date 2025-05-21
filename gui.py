import threading
import time
import customtkinter as ctk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GAApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GA Evolution GUI")
        self.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._create_main_menu()

    def _create_main_menu(self):
        # Main menu frame
        self.menu_frame = ctk.CTkFrame(self)
        self.menu_frame.pack(fill="both", expand=True)

        # Central container
        container = ctk.CTkFrame(self.menu_frame, fg_color="transparent")
        container.pack(expand=True, anchor="n", pady=20)

        # Configure grid
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=0)

        # Add menu items
        title_label = ctk.CTkLabel(container, text="Magic Square - GA Evolution", font=title_font, anchor="center")
        title_label.grid(row=0, column=0, padx=(0,10), pady=30, sticky="n", columnspan=2)

        dimension_label = ctk.CTkLabel(container, text="Magic Square Size:", font=global_font)
        dimension_label.grid(row=1, column=0, padx=(0, 10), pady=20, sticky="w")

        dimension_entry = ctk.CTkEntry(container, width=120, font=global_font)
        dimension_entry.insert(0, '3')
        setattr(self, f"entry_{'N:'.strip(':').lower().replace(' ', '_')}", dimension_entry)
        dimension_entry.grid(row=1, column=1, pady=20, sticky="w")

        max_gen_label = ctk.CTkLabel(container, text="Max Generations:", font=global_font)
        max_gen_label.grid(row=2, column=0, padx=(0, 10), pady=20, sticky="w")

        max_gen_entry = ctk.CTkEntry(container, width=120, font=global_font)
        max_gen_entry.insert(0, '1000')
        setattr(self, f"entry_{'Max Generations:'.strip(':').lower().replace(' ', '_')}", max_gen_entry)
        max_gen_entry.grid(row=2, column=1, pady=20, sticky="w")

        res_type_label = ctk.CTkLabel(container, text="Result Type:", font=global_font)
        res_type_label.grid(row=3, column=0, padx=(0, 10), pady=20, sticky="w")

        res_type_entry = ctk.CTkComboBox(container, values=["regular", "perfect"], width=140, font=global_font)
        res_type_entry.set("regular")
        self.combo_type = res_type_entry
        res_type_entry.grid(row=3, column=1, pady=20, sticky="w")

        # Start button - sends parameters to game screen and initiates GA.
        start_button = ctk.CTkButton(container, text="Start", font=global_font, command=self._on_start, height=50)
        start_button.grid(row=4, column=0, columnspan=2, pady=(30,0))

    def _on_start(self):
        self.menu_frame.pack_forget()
        self._create_gameplay_screen()

        # # TODO: place thread call in gameplay screen.
        # threading.Thread(target=self._run_ga, daemon=True).start()

    # TODO: should start ga calculation from basic_alg.py.
    # TODO: display loading... while ga is working. Display initial configs(?).
    # TODO: Display end result + initial best (?) and graphs and stats below.
    def _create_gameplay_screen(self):
        self.play_frame = ctk.CTkFrame(self)
        self.play_frame.pack(fill="both", expand=True)

        self.initial_stats_frame = ctk.CTkFrame(self.play_frame, fg_color="transparent")
        self.initial_stats_frame.grid(row=0, column=0, padx=(200,50), pady=20, sticky="nw")

        self.final_stats_frame = ctk.CTkFrame(self.play_frame, fg_color="transparent")
        self.final_stats_frame.grid(row=0, column=1, padx=(50, 200), pady=20, sticky="ne")

        self.initial_square_frame = ctk.CTkFrame(self.initial_stats_frame, fg_color="transparent")
        self.initial_square_frame.pack(padx=20, pady=20, anchor="w")

        size = 5
        cell_size = 60

        for i in range(size):
            for j in range(size):

                cell = ctk.CTkFrame(
                    self.initial_square_frame,
                    width=cell_size,
                    height=cell_size,
                    fg_color=("#000000", "#000000"),
                    corner_radius=0,
                    border_width=10
                )
                cell.grid(row=i, column=j, sticky="n")

                num = i * size + j + 1
                lbl = ctk.CTkLabel(
                    cell,
                    text=str(num),
                    text_color="white",
                    fg_color="gray30",
                    font=("Arial", 16),
                    width=cell_size,
                    height=cell_size
                )
                lbl.pack(fill="both", expand=True, padx=1, pady=1)

        self.final_square_frame = ctk.CTkFrame(self.final_stats_frame, fg_color="transparent")
        self.final_square_frame.pack(fill="both", expand=True, padx=20, pady=20, anchor="e")

        size = 5
        cell_size = 60

        for i in range(size):
            for j in range(size):
                cell = ctk.CTkFrame(
                    self.final_square_frame,
                    width=cell_size,
                    height=cell_size,
                    fg_color=("#000000", "#000000"),
                    corner_radius=0,
                    border_width=10
                )
                cell.grid(row=i, column=j, sticky="n")

                num = i * size + j + 1
                lbl = ctk.CTkLabel(
                    cell,
                    text=str(num),
                    text_color="white",
                    fg_color="gray30",
                    font=("Arial", 16),
                    width=cell_size,
                    height=cell_size
                )
                lbl.pack(fill="both", expand=True, padx=1, pady=1)

        self.initial_stats_label = ctk.CTkLabel(
            self.initial_stats_frame,
            text="Initial Stats",
            anchor="center",
            font=("Verdana-Bold", 24)
        )
        self.initial_stats_label.pack(padx=100, pady=6, anchor="w")

        self.final_stats_label = ctk.CTkLabel(
            self.final_stats_frame,
            text="Final Stats",
            anchor="center",
            font=("Verdana-Bold", 24)
        )
        self.final_stats_label.pack(padx=100, pady=6, anchor="w")

        self.initial_fitness_max_label = ctk.CTkLabel(
            self.initial_stats_frame,
            text="Max Fitness: 10 (demo)",
            anchor="center", font=global_font
        )
        self.initial_fitness_max_label.pack(padx=30, pady=6, anchor="w")

        self.initial_fitness_min_label = ctk.CTkLabel(
            self.initial_stats_frame,
            text="Min Fitness: 2 (demo)",
            anchor="center", font=global_font
        )
        self.initial_fitness_min_label.pack(padx=30, pady=6, anchor="w")

        self.final_fitness_best_label = ctk.CTkLabel(
            self.final_stats_frame,
            text="Best Fitness: 100 (demo)",
            anchor="center", font=global_font
        )
        self.final_fitness_best_label.pack(padx=30, pady=6, anchor="w")

        self.final_fitness_min_label = ctk.CTkLabel(
            self.final_stats_frame,
            text="Min Fitness: 43 (demo)",
            anchor="center", font=global_font
        )
        self.final_fitness_min_label.pack(padx=30, pady=6, anchor="w")


title_font = ("Helvetica", 36)
global_font = ("Verdana", 24)

if __name__ == "__main__":
    app = GAApp()
    app.mainloop()