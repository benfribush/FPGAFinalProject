import bnn
import cv2
from PIL import Image as PIL_Image
from PIL import ImageEnhance
from PIL import ImageOps
import imageio 
import numpy as np
import math
from scipy import misc
from array import *
import sys # to access the system
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



print(bnn.available_params(bnn.NETWORK_LFCW1A2))
hw_classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"mnist",bnn.RUNTIME_HW)
sw_classifier = bnn.LfcClassifier(bnn.NETWORK_LFCW1A2,"mnist",bnn.RUNTIME_SW)
print(hw_classifier.classes)

# says we capture an image from a webcam
cap = cv2.VideoCapture(-1) 
_ , cv2_im = cap.read()
cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)

cv2.imwrite('Shadowed.jpg', cv2_im)

yes = cv2.imread('Shadowed.jpg', -1)

rgb_planes = cv2.split(yes)

result_planes = []
result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
    
result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)

cv2.imwrite('shadows_out.jpg', result)
cv2.imwrite('shadows_out_norm.jpg', result_norm)





orig_img_path = '/home/xilinx/jupyter_notebooks/bnn/shadows_out_norm.jpg'
img = PIL_Image.open(orig_img_path).convert("L")     
                   


#img = PIL_Image.fromarray(cv2_im).convert("L") 


#original captured image
#orig_img_path = '/home/xilinx/jupyter_notebooks/bnn/pictures/webcam_image_mnist.jpg'
#img = PIL_Image.open(orig_img_path).convert("L")     
                   
#Image enhancement                
contr = ImageEnhance.Contrast(img)
img = contr.enhance(4.5)                                                    # The enhancement values (contrast and brightness) 
bright = ImageEnhance.Brightness(img)                                     # depends on backgroud, external lights etc
img = bright.enhance(2.0)          

#img = img.rotate(180)                                                     # Rotate the image (depending on camera orientation)
#Adding a border for future cropping
img = ImageOps.expand(img,border=80,fill='white') 




inverted = ImageOps.invert(img)  
box = inverted.getbbox()  
img_new = img.crop(box)  
width, height = img_new.size  
ratio = min((28./height), (28./width))  
background = PIL_Image.new('RGB', (28,28), (255,255,255))  
if(height == width):  
    img_new = img_new.resize((28,28))  
elif(height>width):  
    img_new = img_new.resize((int(width*ratio),28))  
    background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
else:  
    img_new = img_new.resize((28, int(height*ratio)))  
    background.paste(img_new, (int((28-img_new.size[0])/2),int((28-img_new.size[1])/2)))  
  
background  
img_data=np.asarray(background)  
img_data = img_data[:,:,0]  
imageio.imwrite('SMALLimg_webcam_mnist.png', img_data) 
nope = mpimg.imread('SMALLimg_webcam_mnist.png')
#plt.imshow(nope)
#plt.show()



from array import *
from PIL import Image as PIL_Image
from PIL import ImageOps
img_load = PIL_Image.open('SMALLimg_webcam_mnist.png').convert("L")  
# Convert to BNN input format  
# The image is resized to comply with the MNIST standard. The image is resized at 28x28 pixels and the colors inverted.   
  
#Resize the image and invert it (white on black)  
smallimg = ImageOps.invert(img_load)  
smallimg = smallimg.rotate(0)  
  
data_image = array('B')  
  
pixel = smallimg.load()  
#for x in range(0,28):  
#    for y in range(0,28):  
#        if(pixel[y,x] >= 125):  
#            data_image.append(255)  
#        else:  
#            data_image.append(1)  

for x in range(0,28):  
    for y in range(0,28): 
        data_image.append(pixel[y,x])


          
# Setting up the header of the MNIST format file - Required as the hardware is designed for MNIST dataset         
hexval = "{0:#0{1}x}".format(1,6)  
header = array('B')  
header.extend([0,0,8,1,0,0])  
header.append(int('0x'+hexval[2:][:2],16))  
header.append(int('0x'+hexval[2:][2:],16))  
header.extend([0,0,0,28,0,0,0,28])  
header[3] = 3 # Changing MSB for image data (0x00000803)  

pure = data_image
pure = np.frombuffer(data_image, dtype=np.uint8)
pure = pure.reshape(1,-1)
pure = pure.reshape(28,28)
cv2.imwrite('FLIPPEDSMALLimg_webcam_mnist.png', pure) 
data_image = header + data_image  
output_file = open('/home/xilinx/img_webcam_mnist_processed', 'wb')  
data_image.tofile(output_file)  
output_file.close()   


crack = mpimg.imread('FLIPPEDSMALLimg_webcam_mnist.png')
#plt.imshow(crack)
#plt.show()



class_out = hw_classifier.classify_mnist("/home/xilinx/img_webcam_mnist_processed")
print("Class number: {0}".format(class_out))
print("Class name: {0}".format(hw_classifier.class_name(class_out)))

cap.release()

#plt.imshow(nope)
#plt.show()
plt.imshow(crack)
plt.show()
img

