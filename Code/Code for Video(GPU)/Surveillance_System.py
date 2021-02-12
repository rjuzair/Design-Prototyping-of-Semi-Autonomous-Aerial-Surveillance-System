from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2, os, imutils
from main import main
from imutils import paths

def call_main():
    global Map
    threshold = thresh_Slider.get()
    text1 = text.get()
    Map = cv2.imread(text1)
    text2 = folder_path.get()
    Video = cv2.VideoCapture(text2)
    win.destroy()
    main(threshold, Video, Map)
    GUI()
    
def browse_image():
    global folder_path
    filename1 = filedialog.askopenfilename(filetypes=[("Video File",'.mp4')])
    folder_path.set(filename1)

def browse_map():
    global text
    filename1 = filedialog.askopenfilename(filetypes=[("Image File",'.jpg')])
    text.set(filename1)

def GUI():
    global win
    global text
    global folder_path
    global thresh_Slider
    win = Tk()
    text = StringVar()
    folder_path = StringVar()
    win.title("Surveillance System")
    win.geometry('630x630')

    C = Canvas(win)
    background_image = PhotoImage(file = "background.png")
    background_label = Label(win, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
    button1 = Button(win, text = "Process", width = 15, bg = 'Black', fg = 'white', command = call_main)
    button1.place(relx=0.5, rely=0.9, anchor="c")

    entry = Entry(win, textvariable = folder_path, bg = 'Black', fg = 'white', width = 40)
    entry.place(relx=0.38, rely=0.25, anchor="c")

    label = Label(win, text = "Path to Video", fg = 'Black', bg = 'white', width = 10)
    label.place(relx=0.10, rely=0.25, anchor="c")

    entry2 = Entry(win, textvariable = text, bg = 'Black', fg = 'white', width = 40)
    entry2.place(relx=0.38, rely=0.3, anchor="c")

    label2 = Label(win, text = "Path to Map", fg = 'Black', bg = 'white', width = 10)
    label2.place(relx=0.10, rely=0.3, anchor="c")

    button2 = Button(text="Browse Video", width = 20, bg = 'Black', fg = 'white', command = browse_image)
    button2.place(relx=0.8, rely=0.25, anchor="c")

    button3 = Button(text="Browse Map", width = 20, bg = 'Black', fg = 'white', command = browse_map)
    button3.place(relx=0.8, rely=0.3, anchor="c")

    thresh_Slider = Scale(win, from_=0, to=100, bg = 'Black', fg = 'white', orient=HORIZONTAL)
    thresh_Slider.place(relx=0.5, rely=0.72, anchor="c")

    label3 = Label(win, text = "Difference\nThreshold", fg = 'Black', bg = 'white', width = 10)
    label3.place(relx=0.5, rely=0.8, anchor="c")

    win.mainloop()

if __name__ == '__main__':
    GUI()
