import numpy as np

def blackbox_noise(img):

    '''
    With probability od 50% draws area and location black box. 
    ---
    Attributes:
    * img (numpy.ndarray): input image
    ---
    Returns:
    * modified image (numpy.ndarray)
    '''
    if np.random.rand() < 0.5:
        return img
    h, w = img.shape
    blackbox_w = np.random.randint(w // 10, w // 3)
    blackbox_h = np.random.randint(h // 10, h // 3)

    blackbox_x = np.random.randint(0, w)
    blackbox_y = np.random.randint(0, h)

    blackbox_w = min(blackbox_w, w-blackbox_x)
    blackbox_h = min(blackbox_h, h-blackbox_y)

    img[blackbox_y:blackbox_y+blackbox_h, 
        blackbox_x:blackbox_x+blackbox_w] = 0
    
    return img