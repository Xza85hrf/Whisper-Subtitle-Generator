# GUI script
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
# from Generate_TL_Sub_Frm_Video import main, stop_flag
from Generate_TL_Sub_Frm_Video import main

supported_languages = {
    'English': 'en',
    'German': 'de',
    'French': 'fr',
    'Dutch': 'nl',
    'Italian': 'it',
    'Spanish': 'es',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Arabic': 'ar',
    'Persian': 'fa',
    'Japanese': 'ja',
    'Korean': 'ko',
    # Add other languages here...
}


def open_file():
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not filename:
        return
    language_name = language_combo.get()  # Get the language name from the combo box
    if not language_name:
        messagebox.showerror("Error", "Please select a language.")
        return
    language_code = supported_languages[language_name]  # Convert the name to a code
    output_format = format_combo.get()  # Get the output format from the combo box
    output_file = filedialog.asksaveasfilename(
        defaultextension=output_format)  # Ask the user to specify the output file
    if not output_file:
        return
    try:
        threading.Thread(target=main, args=(
            filename, language_code, output_file, int(chunk_size_entry.get()), use_gpu.get())).start()
        result_label.config(text="Subtitle generation started...")
    except Exception as e:
        result_label.config(text=f"Error: {e}")


def stop_process():
    stop_flag.set()


root = tk.Tk()
root.title("Subtitle Generator")
root.geometry("400x200")  # Set the size of the window
root.resizable(True, True)  # Make the window resizable

# Add combo box for language selection
language_label = tk.Label(root, text="Language:")
language_label.pack()
language_combo = ttk.Combobox(root, values=list(supported_languages.keys()))
language_combo.pack()

# Add combo box for output format selection
format_label = tk.Label(root, text="Output Format:")
format_label.pack()
format_combo = ttk.Combobox(root, values=[".srt", ".vtt"])
format_combo.pack()

# Add a check button for using the GPU
use_gpu = tk.BooleanVar()
use_gpu_check = tk.Checkbutton(root, text="Use GPU", variable=use_gpu)
use_gpu_check.pack()

# Add an entry field for the chunk size
chunk_size_label = tk.Label(root, text="Chunk Size:")
chunk_size_label.pack()
chunk_size_entry = tk.Entry(root)
chunk_size_entry.pack()

open_button = tk.Button(root, text="Open Video File", command=open_file)
open_button.pack()

stop_button = tk.Button(root, text="Stop", command=stop_process)
stop_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

progress = ttk.Progressbar(root, length=100, mode='indeterminate')
progress.pack()

root.mainloop()
