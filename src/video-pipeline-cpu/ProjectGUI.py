from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2
from main import main
from imutils import paths

def call_main():
    
    text1 = text.get()
    Map = cv2.imread(text1)
    text2 = folder_path.get()
    Video = cv2.VideoCapture(text2)

    main(Video, Map)
    return

def browse_image():
    global folder_path
    filename1 = filedialog.askopenfilename(filetypes=[("Video File",'.mp4')])
    folder_path.set(filename1)

def browse_map():
    global text
    filename1 = filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    text.set(filename1)

win = Tk()
text = StringVar()
folder_path = StringVar()
win.title("Surveillance System")
win.geometry('500x500')

    
button1 = ttk.Button(win, text = "Load", width = 15, command = call_main)
button1.place(relx=0.5, rely=0.9, anchor="c")

entry = Entry(win,textvariable = folder_path, width = 40)
entry.place(relx=0.28, rely=0.1, anchor="c")

label = Label(win, text = "Path to Video", width = 40)
label.place(relx=0.28, rely=0.05, anchor="c")

entry2 = Entry(win,textvariable = text, width = 40)
entry2.place(relx=0.28, rely=0.2, anchor="c")

label2 = Label(win, text = "Path to Map", width = 40)
label2.place(relx=0.28, rely=0.15, anchor="c")

button2 = Button(text="Browse Video", width = 20, command = browse_image)
button2.place(relx=0.8, rely=0.1, anchor="c")

button3 = Button(text="Browse Map", width = 20, command = browse_map)
button3.place(relx=0.8, rely=0.2, anchor="c")

win.mainloop()
