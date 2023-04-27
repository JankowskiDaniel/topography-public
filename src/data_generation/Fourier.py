import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def get_mask(s, div):
    mask = np.zeros(s, np.float32)
    return cv.circle(mask,(int(s[1]/2), int(s[0]/2)), max(s[0]//div,s[1]//div), 1, -1)
def ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)
img = cv.imread('data/raw/23899.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
m1 = get_mask(img.shape, 10)
m2 = get_mask(img.shape, 20)
m3 = get_mask(img.shape, 30)

img_back_1 = ifft(fshift * m1)
img_back_2 = ifft(fshift * m2)
img_back_3 = ifft(fshift * m3)

magnitude_spectrum = 20*np.log(np.abs(fshift))

images = np.concatenate([img,magnitude_spectrum, magnitude_spectrum *m1 , img_back_1],1)


r,c,i = 4,2,1

plt.subplot(r,c,1)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.subplot(422)
plt.imshow(img, cmap = 'gray')

# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(423)
plt.imshow(magnitude_spectrum*m1, cmap = 'gray')
plt.subplot(424)
plt.imshow(img_back_1, cmap = 'gray')

plt.subplot(425)
plt.imshow(magnitude_spectrum*m2, cmap = 'gray')
plt.subplot(426)
plt.imshow(img_back_2, cmap = 'gray')

plt.subplot(427)

plt.imshow(magnitude_spectrum*m3, cmap = 'gray')
plt.subplot(428)
plt.imshow(img_back_3, cmap = 'gray')

plt.show()
