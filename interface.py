def getfile():
    filename = askopenfilename() 
    print(filename)
    
from tkinter import *
from tkinter.filedialog import askopenfilename

window = Tk()
window.title("Home Number Recognition")
window.geometry("600x600")

topFrame = Frame(window)
topFrame.pack()

bottomframe = Frame(window)
bottomframe.pack( side = BOTTOM )

button = Button(topFrame, text="Choose File", fg="black", command=getfile)
button.pack( side = BOTTOM)

window.mainloop()