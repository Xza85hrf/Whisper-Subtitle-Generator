# GUI script
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
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
        # Keyword args to avoid positional mis-binding against main()'s
        # signature (which currently is input_file, language, output_file,
        # languages, chunk_size, multi_processing, single_chunk_test,
        # model_name).
        chunk_size_raw = chunk_size_entry.get().strip()
        chunk_size = int(chunk_size_raw) if chunk_size_raw else 10
        threading.Thread(
            target=main,
            kwargs=dict(
                input_file=filename,
                language=language_code,
                output_file=output_file,
                chunk_size=chunk_size,
            ),
        ).start()
        result_label.config(text="Subtitle generation started...")
    except Exception as e:
        result_label.config(text=f"Error: {e}")


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

# The transcription worker runs in a daemon thread with no cooperative
# cancel point, so a working Stop button would require the underlying
# module to expose a shared stop_flag. It does not today — the button
# used to reference an undefined name and crashed on click. Close the
# window to terminate.
result_label = tk.Label(root, text="")
result_label.pack()

progress = ttk.Progressbar(root, length=100, mode='indeterminate')
progress.pack()

root.mainloop()
