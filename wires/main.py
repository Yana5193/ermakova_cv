import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image=np.load("C:/Users/anaer/OneDrive/Desktop/учеба/ИГУ/computer_vision/wires/wires3.npy")
struct=np.ones((3,1))
process=binary_opening(image,struct)

labeled_image=label(image)
labeled_process=label(process)
print(f"Original{np.max(labeled_image)}")
print(f"Processed {np.max(labeled_process)}")
num_wires = np.max(labeled_image)
for wire_num in range(1, num_wires + 1):
    current = (labeled_image == wire_num)
    n=binary_opening(current,struct)
    parts = np.max(label(n))
    print(f"Провод {wire_num} порван на {parts} частей")
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(process)
plt.show()
