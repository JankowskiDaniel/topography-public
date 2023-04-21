import random
import cv2
import numpy as np
def draw_blobs(img, SPRAY_PARTICLES = None,SPRAY_DIAMETER = None, fringes_color = None, range_of_blobs = (30,40)):
    n_of_blobs = random.randint(*range_of_blobs)
    i = 0
    w,h = img.shape[0],img.shape[1]
    m_x, m_y = w/2,h/2

    if SPRAY_PARTICLES == None:
       SPRAY_PARTICLES = w*h/150 #640,480 => 2048
    if SPRAY_DIAMETER == None:
        SPRAY_DIAMETER = int((w+h)/100) #640, 480 =>2048
    if fringes_color == None:
        fringes_color = np.min(img) + 10 #80 => 90
        
    
    #n PAPIERZE, 180$, 20 MINUT, I ONE SIĘ NIE PRZYPIEĄ, CZYLI PO TYCH 20 MINU TURBO GRILL I WTEDY KONTROLUJ (OK.5 MIN), ,PRZEWRÓĆ NA DRUGĄ STRONĘ I JESZCZE RAZ TURBO GRILL (OK.2 MIN)
    while i < n_of_blobs:
        
        x=int(random.gauss(m_x,m_x))
        while x>=w or x<0:
            x=int(random.gauss(m_x,m_x))

        y=int(random.gauss(m_y,m_y))
        while y>=h or y<0:
           y=int(random.gauss(m_y,m_y))

        color = np.asscalar(img[x][y])
        if color<90:
            i+=1
            pass
        else:
            continue
        
        coef  = (1-np.sqrt(((x - m_x)/w)**2 + ((y - m_y)/h)**2))
        blob_size = SPRAY_DIAMETER*coef
        blob_density = int(SPRAY_PARTICLES*coef)
        print(SPRAY_DIAMETER,blob_size)

        for n in range(blob_density):
                xo = int(random.gauss(x, blob_size))
                yo = int(random.gauss(y, blob_size))
                if not(( xo >= img.shape[0]) or (yo >= img.shape[1])):
                    img[xo,yo]= int((img[xo,yo]+color)/2)
    print(m_x, m_y, img.shape)
    return img

    

if __name__ == "__main__":
    img = cv2.imread('./src/img.png',cv2.IMREAD_GRAYSCALE)
    print(img.shape) # (480, 640)
    SPRAY_PARTICLES = 900
    SPRAY_DIAMETER = 8
    new_img = draw_blobs(img)
    while True:
        cv2.imshow("img", img)
        if cv2.waitKey(1)==32: 
            break
    cv2.destroyAllWindows()
    pass