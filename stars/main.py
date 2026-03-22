import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.morphology import opening
stars = np.load('C:/Users/anaer/OneDrive/Desktop/учеба/ИГУ/computer_vision/stars/stars.npy')
krest=np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]
])
plus=np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
])
kr=opening(stars,krest)
pl=opening(stars,plus)
labeled_krest=label(kr)
labeled_plus=label(pl)

count_krest=np.max(labeled_krest)
count_plus=np.max(labeled_plus)
count=count_krest+count_plus
all_star=kr+pl
print(f"Количество звезд:{count}")
plt.subplot(121)
plt.imshow(stars)
plt.subplot(122)
plt.imshow(all_star)
plt.show()
plt.show()

