import threading
import tkinter

from basic_alg import genetic_algorithm
import customtkinter as ctk
from tkinter import ttk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_square(dimension, matrix, master):
    cell_size = 300 // dimension

    for i in range(dimension):
        for j in range(dimension):
            cell = ctk.CTkFrame(
                master,
                width=cell_size,
                height=cell_size,
                fg_color=("#000000", "#000000"),
                corner_radius=0,
                border_width=10
            )
            cell.grid(row=i, column=j, sticky="n")

            num = matrix[i * dimension + j]
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

class GAApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GA Evolution GUI")
        self.geometry("1200x950")
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
        dimension_entry.insert(0, '4')
        setattr(self, f"entry_{'N:'.strip(':').lower().replace(' ', '_')}", dimension_entry)
        dimension_entry.grid(row=1, column=1, pady=20, sticky="w")

        max_gen_label = ctk.CTkLabel(container, text="Max Generations:", font=global_font)
        max_gen_label.grid(row=2, column=0, padx=(0, 10), pady=20, sticky="w")

        max_gen_entry = ctk.CTkEntry(container, width=120, font=global_font)
        max_gen_entry.insert(0, '5000')
        setattr(self, f"entry_{'Max Generations:'.strip(':').lower().replace(' ', '_')}", max_gen_entry)
        max_gen_entry.grid(row=2, column=1, pady=20, sticky="w")

        res_type_label = ctk.CTkLabel(container, text="Result Type:", font=global_font)
        res_type_label.grid(row=3, column=0, padx=(0, 10), pady=20, sticky="w")

        res_type_entry = ctk.CTkComboBox(container, values=["Regular", "Perfect"], width=140, font=global_font)
        res_type_entry.set("regular")
        self.combo_type = res_type_entry
        res_type_entry.grid(row=3, column=1, pady=20, sticky="w")

        optimization_method = tkinter.StringVar(value="classic")

        optimize_type_label = ctk.CTkLabel(container, text="Optimization Type:", font=global_font)
        optimize_type_label.grid(row=4, column=0, padx=(0, 10), pady=20, sticky="w")

        classic_radio = ctk.CTkRadioButton(container, text="None", value="classic", variable=optimization_method)
        lamarck_radio = ctk.CTkRadioButton(container, text="Lamarck", value="lamarck", variable=optimization_method)
        darwin_radio = ctk.CTkRadioButton(container, text="Darwin", value="darwin", variable=optimization_method)

        classic_radio.grid(row=4, column=1, padx=(10, 10), pady=25, sticky="w")
        lamarck_radio.grid(row=4, column=1, padx=(110, 10), pady=25, sticky="w")
        darwin_radio.grid(row=4, column=1, padx=(220, 10), pady=25, sticky="w")

        # Start button - sends parameters to game screen and initiates GA.
        start_button = ctk.CTkButton(
            container,
            text="Start",
            font=global_font,
            height=50,
            command=lambda: self.create_gameplay_screen(
                int(dimension_entry.get()),
                int(max_gen_entry.get()),
                res_type_entry.get(),
                optimization_method.get()
            )
        )
        start_button.grid(row=6, column=0, columnspan=2, pady=(30,0))

    # TODO: should start ga calculation from basic_alg.py.
    # TODO: display loading... while ga is working. Display initial configs(?).
    # TODO: Display end result + initial best (?) and graphs and stats below.
    def create_gameplay_screen(self, n: int, generations: int, result_type: str="regular", optimization_method="classic"):

        # Remove main menu
        self.menu_frame.pack_forget()

        play_frame = ctk.CTkFrame(self)
        play_frame.pack(fill="both", expand=True)

        initial_stats_frame = ctk.CTkFrame(play_frame, fg_color="transparent")
        initial_stats_frame.grid(row=0, column=0, padx=(200,50), pady=20, sticky="nw")

        final_stats_frame = ctk.CTkFrame(play_frame, fg_color="transparent")
        final_stats_frame.grid(row=0, column=1, padx=(50, 200), pady=20, sticky="ne")

        results = genetic_algorithm(n=n, generations=generations)
        first_generation = next(results)
        print(first_generation['best_gen1'])

        initial_square_frame = ctk.CTkFrame(initial_stats_frame, fg_color="transparent")
        initial_square_frame.pack(padx=20, pady=20, anchor="w")

        draw_square(n, first_generation['best_gen1'], initial_square_frame)

        final_square_frame = ctk.CTkFrame(final_stats_frame, fg_color="transparent")
        final_square_frame.pack(fill="both", expand=True, padx=20, pady=20, anchor="e")

        draw_square(n, first_generation['best_gen1'], final_square_frame)

        initial_fitness_max_label = ctk.CTkLabel(
            initial_stats_frame,
            text=f"Best Fitness Score: {first_generation['best_score']}",
            anchor="center", font=global_font
        )
        initial_fitness_max_label.pack(padx=30, pady=6, anchor="w")

        final_fitness_best_label = ctk.CTkLabel(
            final_stats_frame,
            text=f"Best Fitness Score: {first_generation['best_score']}",
            anchor="center", font=global_font
        )
        final_fitness_best_label.pack(padx=30, pady=6, anchor="w")

        meta_stats_frame = ctk.CTkFrame(play_frame, fg_color="transparent")
        meta_stats_frame.grid(row=1, column=0, columnspan=2, padx=(50,50), pady=20, sticky="nw")

        meta_stats_label = ctk.CTkLabel(meta_stats_frame, text="Meta Stats", anchor="center", font=global_font)
        meta_stats_label.pack(padx=100, pady=6, anchor="w")

        # Create frames for stats and graphs
        stats_container = ctk.CTkFrame(meta_stats_frame, fg_color="transparent")
        stats_container.pack(fill="x", padx=20, pady=10)

        graphs_container = ctk.CTkFrame(meta_stats_frame, fg_color="transparent")
        graphs_container.pack(fill="x", padx=20, pady=10)

        # Initialize data collection
        self.generation_data = {
            'generation': [],
            'best_scores': [],
            'avg_scores': [],
            'eval_calls': []
        }

        # Create stats labels
        self.avg_best_score_label = ctk.CTkLabel(
            stats_container,
            text="Average Best Score: --",
            anchor="w",
            font=("Verdana", 16)
        )
        self.avg_best_score_label.grid(row = 0, column = 0, padx=20, pady=5, sticky="w")

        self.total_evals_label = ctk.CTkLabel(
            stats_container,
            text="Total Evaluations: --",
            anchor="w",
            font=("Verdana", 16)
        )
        self.total_evals_label.grid(row = 1, column = 0, padx=20, pady=5, sticky="w")

        self.convergence_rate_label = ctk.CTkLabel(
            stats_container,
            text="Convergence Rate: --",
            anchor="w",
            font=("Verdana", 16)
        )
        self.convergence_rate_label.grid(row = 2, column = 0, padx=20, pady=5, sticky="w")

        solution_found_label = ctk.CTkLabel(
            stats_container,
            text="Solution Found: --",
            anchor="w",
            font=("Verdana", 16)
        )
        solution_found_label.grid(row = 0, column = 1, padx=20, pady=5, sticky="w")

        # Create figure for graphs
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graphs_container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update()

        for checkpoint in results:
            # Update square display
            for child in final_square_frame.winfo_children():
                child.destroy()
            draw_square(n, checkpoint['best_solution'], final_square_frame)

            # Update fitness label
            final_fitness_best_label.configure(text=f'Best Fitness Score: {checkpoint["best_score"]}')

            # Collect data for statistics
            self.generation_data['generation'].append(checkpoint['generation'])
            self.generation_data['best_scores'].append(checkpoint['best_score'])
            self.generation_data['avg_scores'].append(checkpoint['avg_scores'])
            self.generation_data['eval_calls'].append(checkpoint['eval_calls'])

            # Update statistics
            avg_best = np.mean(self.generation_data['best_scores'])
            total_evals = checkpoint['eval_calls']
            convergence_rate = len(set(self.generation_data['best_scores'][-10:])) / 10 if len(
                self.generation_data['best_scores']) >= 10 else 0

            self.avg_best_score_label.configure(text=f"Average Best Score: {avg_best:.2f}")
            self.total_evals_label.configure(text=f"Total Evaluations: {total_evals}")
            self.convergence_rate_label.configure(text=f"Convergence Rate: {convergence_rate:.2f}")

            # Update graphs
            self.ax1.clear()
            self.ax2.clear()

            # Create DataFrame with proper numeric values
            df = pd.DataFrame({
                'generation': self.generation_data['generation'],
                'best_scores': self.generation_data['best_scores'],  # Already a single number
                'avg_scores': [np.mean(scores) for scores in self.generation_data['avg_scores']]
            })

            # Plot best scores
            sns.lineplot(data=df, x='generation', y='best_scores', ax=self.ax1)
            self.ax1.set_title('Best Scores Over Generations')
            self.ax1.set_xlabel('Generation')
            self.ax1.set_ylabel('Best Score')

            # Plot average scores
            sns.lineplot(data=df, x='generation', y='avg_scores', ax=self.ax2)
            self.ax2.set_title('Average Scores Over Generations')
            self.ax2.set_xlabel('Generation')
            self.ax2.set_ylabel('Average Score')

            self.update()

            self.fig.tight_layout()
            self.canvas.draw()

            self.update()

        solution_found_label.configure(
            text="Solution Found: Yes" if self.generation_data["best_scores"][-1] == 0 else "Solution Found: No"
        )

        finished_label = ctk.CTkLabel(
            stats_container,
            text="Finished!",
            anchor="w",
            font=("Verdana", 24)
        )
        finished_label.grid(row=1, column=1, padx=20, pady=20, sticky="w")

title_font = ("Helvetica", 18)
global_font = ("Verdana", 16)

if __name__ == "__main__":
    app = GAApp()
    app.mainloop()