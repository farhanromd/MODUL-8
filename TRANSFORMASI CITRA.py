#transformasi Twirl
import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage.transform import swirl

img = cv2.imread ("serigala.jpeg")

swirled = swirl(img, rotation=0, strength=10,
radius=120)
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
figsize=(8, 3),sharex=True,sharey=True)

ax0.imshow(img, cmap=plt.cm.gray)
ax0.axis('off')
ax1.imshow(swirled, cmap=plt.cm.gray)
ax1.axis('off')
plt.show()

#translation
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread ("serigala.jpeg")
h, w = img.shape[:2]
half_height, half_width = (h//4), (w//8)
transition_matrix = np.float32([[1, 0, half_width],[0, 1,half_height]])
img_transition = cv.warpAffine(img, transition_matrix, (w, h))

plt.imshow(cv.cvtColor(img_transition,cv.COLOR_BGR2RGB))
plt.title("Translation")
plt.show()


#rotation
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread ("serigala.jpeg")
h, w = img.shape[:2]

rotation_matrix = cv.getRotationMatrix2D((w/2,h/2), -180, 0.5)

rotated_image = cv.warpAffine(img, rotation_matrix, (w,h))

plt.imshow(cv.cvtColor(rotated_image, cv.COLOR_BGR2RGB))
plt.title("Rotation")
plt.show()


import mahotas as mh
import numpy as np
from pylab import imshow, show

regions = np.zeros((8,8), bool)

regions[:4,:3] = 1
regions[:5,6:] = 1
labeled, nr_objects = mh.label(regions)

imshow(labeled, interpolation='nearest')
show()

labeled,nr_objects = mh.label(regions, np.ones((3,3), bool))
sizes = mh.labeled.labeled_size(labeled)
print('Background size', sizes[0])
print('Size of first region:{}'.format(sizes[1]))

array = np.random.random_sample(regions.shape)
sums = mh.labeled_sum(array, labeled)
print('Sum of first region:{}'.format(sums[1]))


#skala interpolasi miring
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread ("serigala.jpeg")

fig, ax = plt.subplots(1, 3, figsize=(16, 8))

# image size being 0.15 times of it's originalsize
img_scaled = cv.resize(img, None, fx=0.15, fy=0.15)
ax[0].imshow(cv.cvtColor(img_scaled, cv.COLOR_BGR2RGB))
ax[0].set_title("Linear Interpolation Scale")

# image size being 2 times of it's original size
img_scaled_2 = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
ax[1].imshow(cv.cvtColor(img_scaled_2, cv.COLOR_BGR2RGB))
ax[1].set_title("Cubic Interpolation Scale")

# image size being 0.15 times of it's original size
img_scaled_3 = cv.resize(img, (200, 400),interpolation=cv.INTER_AREA)
ax[2].imshow(cv.cvtColor(img_scaled_3, cv.COLOR_BGR2RGB))
ax[2].set_title("Skewed Interpolation Scale")
