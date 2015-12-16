import mnist_loader
from network import *
import numpy as np
from theano import *
import theano.tensor as T
from Tkinter import *
from ttk import *
from math import ceil, floor
import signal
import time
import aggdraw
import matplotlib
import tkFont
from scipy.ndimage.filters import gaussian_filter
  
from PIL import Image,ImageTk

class App:

    def __init__(self, master, guess_fn, sublayer_funcs, feature_maps, data_set = None):
        big_frame = Frame(master, width=100, height=100)
        big_frame.grid(row=1,column=1)
        
        self.guess_fn = guess_fn
        self.sublayer_funcs = sublayer_funcs
        self.data_set = data_set
        
        self.img_data = np.zeros((28*28)).astype(config.floatX)

        self.button = Button(
            big_frame, text="QUIT", command=big_frame.quit
            )
        self.button.pack(side=TOP)
        
        self.inputTxt = StringVar()
        self.inputTxt.trace('w', lambda name, index, mode, sv=self.inputTxt: self.clear_image(None) if sv.get()=='' else self.load_data(int(sv.get())))
        
        self.dataEntry = Entry(big_frame, textvariable=self.inputTxt)
        self.dataEntry.pack(side=TOP)
        
        self.prediction_bars = []
        self.prediction_labels = []
        prediction_bars_frame = Frame(master, borderwidth=20)
        prediction_bars_frame.grid(row=1,column=2)
        self.customFont = tkFont.Font(family="Helvetica", size=16)
        self.heavyFont = tkFont.Font(family="Helvetica", size=22, weight='bold')
        for i in xrange(5):
            self.prediction_labels.append(Label(prediction_bars_frame,text="0", font=self.customFont))
            self.prediction_labels[i].grid(row=i,column=0)
            self.prediction_bars.append(Progressbar(prediction_bars_frame,orient='horizontal',
                                                    length=150,mode='determinate'))
            self.prediction_bars[i].grid(row=i,column=1)
        self.prediction_labels[0].config(font=self.heavyFont)
            
        self.layer_frames = []
        self.sublayer_frame_sets = []
        self.sublayer_label_sets = []
        self.sublayer_images_frame = Frame(master)
        self.sublayer_images_frame.grid(row=1,column=3)
        
        self.arrow_imgs = []
        self.arrow_labels = []
        
        sublayer_conv_images_frame = Frame(self.sublayer_images_frame)
        sublayer_conv_images_frame.pack(side=TOP)
        sublayer_comb_images_frame = Frame(self.sublayer_images_frame)
        sublayer_comb_images_frame.pack(side=BOTTOM)
        
        offset = 0
        for layer_num, sublayer_func in enumerate(self.sublayer_funcs):
            output = sublayer_func(self.img_data)
            layer_frame = Frame(sublayer_conv_images_frame, padding=10, borderwidth=5, relief='groove')
            layer_frame.pack()
            self.layer_frames.append(layer_frame)
            
            if layer_num < len(self.sublayer_funcs)-1:
                cwd = os.path.dirname(os.path.realpath(__file__))
                arrow_img = ImageTk.PhotoImage(Image.open(os.path.join(cwd,'arrow.png')))
                self.arrow_imgs.append(arrow_img)
                arrow_label = Label(sublayer_conv_images_frame, image=arrow_img)
                arrow_label.pack()
                self.arrow_labels.append(arrow_label)
            if output.ndim >= 3: #Convolutional/Pooling Layer
                self.sublayer_frame_sets.append([Frame(layer_frame) 
                                                for x in xrange(output.shape[1])])
                self.sublayer_label_sets.append([Label(small_frame) 
                                                for small_frame in self.sublayer_frame_sets[-1]])
                for (col,small_frame) in enumerate(self.sublayer_frame_sets[-1]):
                    row_width = int(10*ceil((float(layer_num)+1)/2))
                    small_frame.grid(row=layer_num+offset,column=col%row_width)
                    if ((col+1)%row_width == 0):
                        offset += 1
                for label in self.sublayer_label_sets[-1]:
                    label.pack()
            else: # Fully Connected Layer
                self.sublayer_label_sets.append([Label(layer_frame)])
                for label in self.sublayer_label_sets[-1]:
                    label.pack(side=TOP)
            
        self.sublayer_fmap_images = []
        self.sublayer_fmap_labels = []
        self.update_prediction_display()
        
        for sublayer_fmap_set,sublayer_frame_set in zip(feature_maps,self.sublayer_frame_sets[::2]):
            sublayer_fmap_data = sublayer_fmap_set.get_value()
            if sublayer_fmap_data.shape[1] == 1:
                sublayer_fmap_label_set = ([Label(sublayer_frame) 
                                                for sublayer_frame in sublayer_frame_set])
                self.sublayer_fmap_labels.append(sublayer_fmap_label_set)
                max_val = np.abs(np.max(sublayer_fmap_data))
                fmap_imagedata = np.vsplit(sublayer_fmap_data,sublayer_fmap_data.shape[0])
                fmap_imagedata = [np.squeeze(arr) for arr in fmap_imagedata]
                for i,imagedata in enumerate(fmap_imagedata):
                    my_cm = matplotlib.cm.get_cmap('coolwarm')
                    max_val = np.min(imagedata)
                    normed_data = (imagedata + max_val) / (2*max_val)
                    fmap_imagedata[i] = my_cm(normed_data,bytes=True)
                sublayer_fmap_image_set = [ImageTk.PhotoImage(Image.fromarray(img, mode='RGBA').resize([8*x for x in img.transpose().shape[-2:]]))
                                                for img in fmap_imagedata]
                #sublayer_fmap_image_set[-1].show()
                self.sublayer_fmap_images.append(sublayer_fmap_image_set)
                for (fmap_image,label) in zip(sublayer_fmap_image_set,sublayer_fmap_label_set):
                    label.pack(side=BOTTOM)
                    label.config(image=fmap_image)
        
        self.big_img_size = 420
        self.canvas = Canvas(big_frame, width=self.big_img_size, height=self.big_img_size)
        self.canvas.bind("<Button-1>", self.clear_image)
        self.canvas.bind("<B1-Motion>", self.add_pixel)
        self.canvas.pack(side=LEFT)

        #cr.set_source_rgb(0,0,0)
        self.char_img = Image.new('L',(28,28),'white')
        self.big_img = ImageTk.PhotoImage(self.char_img.resize((self.big_img_size,self.big_img_size)))
        self.canvas_img = self.canvas.create_image(self.big_img_size/2,self.big_img_size/2,image=self.big_img)
       
    def load_data(self, index):
        index = index % len(self.data_set)
        self.char_img = Image.fromarray(255*np.subtract(1,self.data_set[index].reshape((28,28)))).convert('L')
        self.update_image_data(user_input=False)
        self.update_prediction_display()
          
    def window2imagecoords(self,x,y):
        x = x*28/self.big_img_size
        y = y*28/self.big_img_size
        return (x,y)
      
    def clear_image(self, event):
        self.char_img = Image.new('L',(28,28),'white')
        self.update_image_data(user_input = False)
        self.update_prediction_display()
        self.last_cursor_loc = (self.window2imagecoords(event.x, event.y))
            
    def update_prediction_display(self):
        guess = np.squeeze(self.guess_fn(self.img_data))
        predicted_nums = np.argsort(guess)[::-1]
        prediction_strengths = np.sort(guess)[::-1]
        for i in xrange(5):
            self.prediction_labels[i].config(text=str(predicted_nums[i]))
            self.prediction_bars[i].config(value=prediction_strengths[i]*100)
        
        self.sublayer_images = []
        self.sublayer_fmap_images = []
        for layer_num, sublayer_func in enumerate(self.sublayer_funcs):
            output = sublayer_func(self.img_data)
            
            if (output.ndim >= 3): #Convolutional/Pooling Layer
                sublayer_images = np.squeeze(np.hsplit(output,output.shape[1]))
                max_val = np.max(sublayer_images)
                sublayer_images = [ImageTk.PhotoImage(Image.fromarray(img*255/max_val, mode='F').convert('L').resize([4*x for x in img.transpose().shape]))
                                            for img in sublayer_images]
            else: # Fully Connected Layer
                if output.size >= 70:
                    output = output.reshape((2,output.size/2))
                sublayer_images = [output]
                sublayer_images = [ImageTk.PhotoImage(Image.fromarray(255*img, mode='F').convert('L').resize([16*x for x in img.transpose().shape]))
                                            for img in sublayer_images]
            
            self.sublayer_images.append(sublayer_images)
            
            for (image,label) in zip(sublayer_images,self.sublayer_label_sets[layer_num]):
                label.config(image=image)
    
    def update_image_data(self, user_input = True):
        if user_input:
            self.filtered_img = gaussian_filter(self.char_img, 0.3)
        else:
            self.filtered_img = gaussian_filter(self.char_img, 0.0)
        big_img = Image.fromarray(self.filtered_img).resize((self.big_img_size,self.big_img_size))
        self.big_img = ImageTk.PhotoImage(big_img)
        self.canvas.itemconfig(self.canvas_img,image=self.big_img)
        self.img_data = (255 - self.filtered_img.reshape(784).astype(config.floatX)) / 255
    
    def add_pixel(self, event):
        curr_loc = self.window2imagecoords(event.x, event.y)
        d = aggdraw.Draw(self.char_img)
        p = aggdraw.Pen("black",2.3,255)
        path = self.last_cursor_loc + curr_loc
        d.line(path,p)
        d.flush()
        self.last_cursor_loc = curr_loc
        
        self.update_image_data()
        self.update_prediction_display()

def save_params_early(net, test_data):
    net.save_params('params-' + time.strftime("%Y%m%d-%H%M%S") + '-' + str(net.evaluate(test_data)))
    sys.exit(0)

def main():
    
    mini_batch_size = 20
    layers = [ConvLayer((20, 1, 5, 5), (mini_batch_size, 1, 28, 28), border_mode='valid', dropout_rate = 0.0, unique_weights_per_input = True, activation_fn=ReLU),
                PoolLayer((mini_batch_size,20,24,24),(2,2)),
                ConvLayer((40, 20, 5, 5), (mini_batch_size, 20, 12, 12), border_mode='valid', dropout_rate = 0.0, unique_weights_per_input = False, activation_fn=ReLU),
                PoolLayer((mini_batch_size,40,8,8),(2,2)),
                FullyConnectedLayer(40*4*4, 100, dropout_rate = 0.5, activation_fn = ReLU),
                FullyConnectedLayer(100, 10, dropout_rate = 0.5, activation_fn = softmax)]
        
    #layers = [FullyConnectedLayer(28*28, 30), FullyConnectedLayer(30,10)]
    #layers = [ConvLayer((1,1,28,19), (mini_batch_size, 1,28, 28))]
    net = Network(layers, mini_batch_size, cost_func = log_likelihood, )
    
    #print net.feedforward(training_data[0][0]), training_data[1][0]
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(samples = 4, trans = 3, angle = 10, scaling = 5)
    print "Training with " + str(len(training_data[0])) + " images..."
    signal.signal(signal.SIGINT, lambda signal, frame: save_params_early(net, test_data))
    
    #net.save_params('params-' + time.strftime("%Y%m%d-%H%M%S") + '-' + str(net.evaluate(test_data)))
    
    net.load_params('params-20151216-032719-9834')
    
    #net.load_feature_maps('params-20151216-013626-9891', 0, [0,2,3,4,5,7,10,12,13,14], 0, range(10))
    #net.load_feature_maps('params-20151216-020041-9847', 0, [2,3,4,5,6,7,12,14,15,17], 0, range(10,20))
    
    #net.SGD(training_data, 40, mini_batch_size, 0.005, l2_lambda = 0.000, test_data = test_data)
    
    
    #net.load_feature_maps('params-20151216-013626-9891', 2, [0,3,4,6,8,9,12,13,15,16,19,22,24,26,27,30,35,37], 2, range(18))
    #net.load_feature_maps('params-20151216-020041-9847', 2, [0,2,8,15,16,19,27,28,36,39], 2, range(18,28))
    
    #net.load_feature_maps('params-20151211-175309-9856', 0, range(0,5), 0, range(5,9))
    #print net.evaluate(test_data)
    #print net.feedforward(training_data[0][0]), training_data[1][0]
    
    root = Tk()
    
    s = Style()
    s.configure('My.TFrame', borderwidth=30)
    
    center_frame = Frame(root)
    center_frame.place(relx=.5, rely=.5, anchor="c")
    app = App(center_frame, net.feedforward, net.out_funcs, net.fmaps, test_data[0])
    
    
    root.mainloop()
    root.destroy()

main()