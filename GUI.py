import tkinter as tk
from tkinter import filedialog
from detection import Detection


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("File Selection App")

        self.selected_file = tk.StringVar(value="No file selected")
        self.show_guitar = tk.IntVar()
        self.show_fingers = tk.IntVar()

        self.create_widgets()

    def create_widgets(self):
        # Select File Button
        select_file_button = tk.Button(self.root, text="Select File", command=self.select_file, width=20)
        select_file_button.grid(row=1, columnspan=2, padx=10, pady=10)

        # Label to display selected file
        self.selected_file_label = tk.Label(self.root, textvariable=self.selected_file, anchor='w')
        self.selected_file_label.grid(row=2, columnspan=2, padx=10, pady=10)

        # Check Button 1
        check_button1 = tk.Checkbutton(self.root, text="Show guitar detection", variable=self.show_guitar)
        check_button1.grid(row=3, pady=5)

        # Check Button 2
        check_button2 = tk.Checkbutton(self.root, text="Show hand markers", variable=self.show_fingers)
        check_button2.grid(row=4, pady=5)

        # Start Button
        start_button = tk.Button(self.root, text="Start", command=self.start, width=20)
        start_button.grid(row=5, columnspan=2, pady=10)

        # Set column weights to center elements along the x-axis
        self.root.columnconfigure(0, weight=1)
        # self.root.columnconfigure(1, weight=1)

        # Set window size
        # self.root.geometry("300x200")
        self.root.minsize(width=300, height=200)

    def start(self):
        # Function to be triggered when "Start" button is clicked
        print("Start function triggered")
        print(f"Option 1 selected: {self.show_guitar.get()}")
        print(f"Option 2 selected: {self.show_fingers.get()}")
        print(f"Selected file: {self.selected_file}")

        test = Detection(self.selected_file.get(), reflection=True)
        test.resize((1200, 600))
        test.show(guitar=self.show_guitar.get(), fingers=self.show_fingers.get())

    def select_file(self):
        # Function to be triggered when "Select File" button is clicked
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("All Files", "*.*")])

        if file_path:
            self.selected_file.set(file_path)
            print(f"Selected file: {file_path}")
