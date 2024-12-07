import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cnn_logic import CNNModel, preprocess_images
import numpy as np
from collections import Counter


class CNNVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("CNN Training Visualization")
        self.create_ui()

        self.X = None
        self.y = None
        self.network = None
        self.loss_values = []
        self.accuracy_values = []

    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        left_frame = ttk.Frame(main_frame, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.S))

        right_frame = ttk.Frame(main_frame, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.E, tk.S))

        controls_frame = ttk.Frame(left_frame, padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.N, tk.W))

        ttk.Label(controls_frame, text="Number of Classes:").grid(row=0, column=0, sticky=tk.W)
        self.output_size = tk.IntVar(value=10)
        ttk.Entry(controls_frame, textvariable=self.output_size).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(controls_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W)
        self.learning_rate = tk.DoubleVar(value=0.001)
        ttk.Entry(controls_frame, textvariable=self.learning_rate).grid(row=1, column=1, sticky=tk.W)

        ttk.Button(controls_frame, text="Load Dataset", command=self.load_images).grid(row=2, column=0, sticky=tk.W)
        ttk.Button(controls_frame, text="Start Training", command=self.start_training_thread).grid(row=2, column=1, sticky=tk.W)
        ttk.Button(controls_frame, text="Visualize Filters", command=self.visualize_filters).grid(row=3, column=0, sticky=tk.W)
        ttk.Button(controls_frame, text="Visualize Feature Maps", command=self.visualize_feature_maps).grid(row=3, column=1, sticky=tk.W)

        self.canvas = tk.Canvas(left_frame, width=400, height=300, bg="white")
        self.canvas.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        plot_frame = ttk.Frame(right_frame, padding="5")
        plot_frame.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E))

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.loss_ax = self.figure.add_subplot(111)
        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.grid()

        self.loss_canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.loss_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)

        log_frame = ttk.Frame(right_frame, padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))

        ttk.Label(log_frame, text="Training Log:").grid(row=0, column=0, sticky=tk.W)

        self.log_text = tk.Text(log_frame, wrap="word", height=10, width=60)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.config(state=tk.DISABLED)

        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text['yscrollcommand'] = log_scroll.set
        log_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))

        self.accuracy_label = ttk.Label(controls_frame, text="")
        self.accuracy_label.grid(row=4, column=0, columnspan=2, sticky=tk.W)

    def log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def load_images(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                self.log_message("Loading dataset...")
                self.X, self.y, input_shape = preprocess_images(folder_path)
                self.output_size.set(self.y.shape[1])
                self.log_message(f"Dataset loaded: {self.X.shape[0]} samples, {self.y.shape[1]} classes")
                self.log_message(f"Class distribution: {Counter(np.argmax(self.y, axis=1))}")
                self.generate_network(input_shape)
                self.draw_network_architecture()
                self.log_message("Network initialized.")
                messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")
            except Exception as e:
                self.log_message(f"Error: {e}")
                messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

    def generate_network(self, input_shape):
        self.network = CNNModel(input_shape=input_shape, num_classes=self.output_size.get())

    def draw_network_architecture(self):
        if not self.network:
            self.log_message("No network to visualize.")
            return

        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        layer_spacing = 150
        node_radius = 20

        x_offset = 50
        y_center = canvas_height // 2

        input_shape = self.network.input_shape
        input_label = f"Input: {input_shape[0]}x{input_shape[1]}x{input_shape[2]}"
        self.canvas.create_oval(
            x_offset - node_radius, y_center - node_radius,
            x_offset + node_radius, y_center + node_radius,
            fill="blue", outline="black"
        )
        self.canvas.create_text(x_offset, y_center + 30, text=input_label)

        x_offset += layer_spacing
        for i, (layer_type, layer_info) in enumerate(self.network.get_layer_info()):
            layer_label = f"{layer_type}: {layer_info}"
            self.canvas.create_rectangle(
                x_offset - 50, y_center - 30,
                x_offset + 50, y_center + 30,
                fill="green" if layer_type == "Conv" else "yellow", outline="black"
            )
            self.canvas.create_text(x_offset, y_center + 50, text=layer_label)
            self.canvas.create_line(
                x_offset - layer_spacing + 50, y_center,
                x_offset - 50, y_center,
                arrow=tk.LAST
            )
            x_offset += layer_spacing

        output_label = f"Output: {self.network.num_classes} classes"
        self.canvas.create_oval(
            x_offset - node_radius, y_center - node_radius,
            x_offset + node_radius, y_center + node_radius,
            fill="red", outline="black"
        )
        self.canvas.create_text(x_offset, y_center + 30, text=output_label)

    def start_training_thread(self):
        if self.X is None or self.y is None:
            self.log_message("Please load a dataset first!")
            return
        threading.Thread(target=self.start_training).start()

    def start_training(self):
        if self.X.size == 0 or self.y.size == 0:
            self.log_message("Invalid dataset. Please load a valid dataset before training.")
            return
        epochs = 100
        learning_rate = self.learning_rate.get()
        self.log_message("Starting training...")
        for epoch in range(epochs):
            try:
                predictions = self.network.forward(self.X)
                loss = self.network.compute_loss(predictions, self.y)
                accuracy = self.network.compute_accuracy(predictions, self.y)
                self.network.backward(self.X, self.y, predictions, learning_rate)
                self.loss_values.append(loss)
                self.accuracy_values.append(accuracy)
                self.log_message(f"Epoch {epoch + 1}/{epochs}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")
                self.update_plots()
                learning_rate *= 0.95
            except Exception as e:
                self.log_message(f"Error during training: {e}")
                break
        self.log_message("Training completed!")

    def update_plots(self):
        self.loss_ax.clear()
        self.loss_ax.set_title("Training Loss")
        self.loss_ax.set_xlabel("Epoch")
        self.loss_ax.set_ylabel("Loss")
        self.loss_ax.plot(self.loss_values, label="Loss", color="blue")
        self.loss_ax.legend()
        self.loss_canvas.draw()

    def visualize_filters(self):
        if self.network is None:
            self.log_message("Network not initialized.")
            return
        filters = self.network.weights['conv']
        num_filters = filters.shape[-1]
        fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
        for i in range(num_filters):
            axes[i].imshow(filters[:, :, 0, i], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Filter {i + 1}")
        plt.show()

    def visualize_feature_maps(self):
        if self.network is None or self.X is None:
            self.log_message("Load a dataset and train the network first.")
            return
        sample = self.X[0:1]
        feature_maps = self.network.convolve(sample, self.network.weights['conv'])
        feature_maps = self.network.relu(feature_maps)
        num_maps = feature_maps.shape[-1]
        fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
        for i in range(num_maps):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"Feature Map {i + 1}")
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = CNNVisualizer(root)
    root.mainloop()
