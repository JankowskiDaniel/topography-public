from AbstractImage import AbstractImage
from AbstractDecorator import AbstractDecorator
import random
import cv2
import numpy as np


def add_bubbles(img, SPRAY_PARTICLES = 800,SPRAY_DIAMETER = 8, fringes_color = None, range_of_blobs = (30,40), sigma=100):
    nr_of_blobs = random.randint(*range_of_blobs)
    i = 0
    w,h = img.shape[0],img.shape[1]
    m_x, m_y = w/2,h/2

    if SPRAY_PARTICLES == None:
       SPRAY_PARTICLES = w*h/150 #640,480 => 2048
    if SPRAY_DIAMETER == None:
        SPRAY_DIAMETER = int((w+h)/100) #640, 480 =>2048
    if fringes_color == None:
        fringes_color = np.min(img) + 10 #80 => 90
    
    while i < nr_of_blobs:
        
        #coordinates of the blob
        x=int(random.gauss(m_x,sigma))

        while x>=w or x<0:
            x=int(random.gauss(m_x,sigma))

        y=int(random.gauss(m_y,sigma))

        while y>=h or y<0:
           y=int(random.gauss(m_y,sigma))

        color = img[x][y].item(0)
        if color<fringes_color:
            i+=1
            pass
        else:
            continue
        
        coef  = (1-np.sqrt(((x - m_x)/w)**2 + ((y - m_y)/h)**2))
        blob_size = SPRAY_DIAMETER*coef
        blob_density = int(SPRAY_PARTICLES*coef)
        for n in range(blob_density):
                xo = int(random.gauss(x, blob_size))
                yo = int(random.gauss(y, blob_size))
                if not(( xo >= img.shape[0]) or (yo >= img.shape[1])):
                    img[xo,yo]= int((img[xo,yo]+color)/2)

    return img

class Bubble(AbstractDecorator):
    """
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    """
    def __init__(self, component, SPRAY_PARTICLES = None,SPRAY_DIAMETER = None, fringes_color = None, range_of_blobs = (30,40), sigma = None) -> None:
        super().__init__(component)
        self.SPRAY_PARTICLES=SPRAY_PARTICLES
        self.SPRAY_DIAMETER=SPRAY_DIAMETER
        self.fringes_color =fringes_color 
        self.range_of_blobs = range_of_blobs


    def generate(self) -> str:
        """
        Decorators may call parent implementation of the generate, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        """
        img = self.component.generate()
        img = add_bubbles(img, self.SPRAY_PARTICLES, self.SPRAY_DIAMETER,  self.fringes_color, self.range_of_blobs)
        return img

