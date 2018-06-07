# Simple enough, just import everything from tkinter.
from tkinter import *

import matplotlib.pyplot as plt

from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):
    image_size = 32
    num_of_channels = 1
    num_classifiers = 6
    prediction_label = -1
    model_saver = -1

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        # parameters that you want to send through the Frame class.
        Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    # Creation of init_window
    def init_window(self):
        # changing the title of our master widget
        self.master.title("Home Number Recognition")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        # added "file" to our menu
        menu.add_cascade(label="File", menu=file)

        topFrame = Frame(self)
        topFrame.pack()

        button = Button(topFrame, text="Choose Image", fg="black", command=self.showImg)
        button.pack(side=BOTTOM)

        self.prediction_label = Label(self, text="", font=("Helvetica", 16))
        self.prediction_label.pack()
        self.model_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')

    def normalization(self, img):
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        img = self.rgb2gray(img)  # RGB to greyscale
        pixel_depth = 255.0
        return (np.array(img, dtype='float32') - (pixel_depth / 2)) / (pixel_depth / 2)

    def rgb2gray(self, img):
        return np.dot(np.array(img, dtype='float32'), [0.299, 0.587, 0.114])

    def formatForFeedForward(self, dataset):
        dataset = np.reshape(dataset,
                             (-1, self.image_size, self.image_size, self.num_of_channels)).astype(np.float32)
        return dataset

    def get_onehot_as_string(self, labels):
        all_labels = []
        batch_size = labels[0].shape[0]
        for i in range(batch_size):
            num_digits = np.argmax(labels[0][i])
            st = str(num_digits)
            for j in range(1, self.num_classifiers):
                if (j > num_digits):
                    break
                st = st + str(np.argmax(labels[j][i]))
            all_labels.append(st)
        return all_labels

    def classifyImages(self, img):
        img = self.formatForFeedForward(img)
        with tf.Session() as sess:
            # new_saver = tf.train.import_meta_graph('./best_model/saved_model/model.ckpt.meta')
            self.model_saver.restore(sess, tf.train.latest_checkpoint('./best_model/saved_model/'))
            graph = sess.graph
            one_input = graph.get_tensor_by_name("Placeholder_8:0")
            softmax1 = graph.get_tensor_by_name("Softmax_18:0")
            softmax2 = graph.get_tensor_by_name("Softmax_19:0")
            softmax3 = graph.get_tensor_by_name("Softmax_20:0")
            softmax4 = graph.get_tensor_by_name("Softmax_21:0")
            softmax5 = graph.get_tensor_by_name("Softmax_22:0")
            softmax6 = graph.get_tensor_by_name("Softmax_23:0")

            # print([node.name for node in graph.as_graph_def().node])

            feed_dict = {one_input: img}
            c1, c2, c3, c4, c5, c6 = sess.run(
                [softmax1, softmax2, softmax3, softmax4, softmax5,
                 softmax6], feed_dict=feed_dict)
            predictions = [c1, c2, c3, c4, c5, c6]
            prediction_str = self.get_onehot_as_string(predictions)
            return prediction_str[0][1:]

    def showImg(self):
        filename = askopenfilename()
        print(filename)
        im = Image.open(filename)
        img_label_size = 200
        resized = im.resize((img_label_size, img_label_size), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(resized)
        im = self.normalization(im)
        # plt.imshow(im)
        # plt.show()
        # labels can be text or images
        im = Image.fromarray(im.astype('float32'))
        img = Label(self, image=render, width=img_label_size, height=img_label_size)
        img.image = render
        img.place(x=200, y=100)
        prediction = self.classifyImages(im)
        self.showText(prediction)

    def showText(self, message="hello"):

        self.prediction_label.config(text="Prediction: " + message)

    def client_exit(self):
        exit()


# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = Tk()

root.geometry("600x400")

# creation of an instance
app = Window(root)

# mainloop
root.mainloop()