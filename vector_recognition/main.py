import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent
  
def count_holes(region):
    shape=region.image.shape 
    new_image=np.zeros((shape[0]+2,shape[1]+2))
    new_image[1:-1,1:-1]=region.image
    new_image=np.logical_not(new_image)
    labeled=label(new_image)
    return np.max(labeled)-1
  
def vertik_symm(region):
    img = region.image.astype(float)
    h, w = img.shape
    if w < 6:
        return 0.0
    mid = w // 2
    left = img[:, :mid]
    right = np.fliplr(img[:, w - mid:])   
    difference = np.mean(np.abs(left - right))
    symmetry = 1.0 - difference
    return symmetry

def extractor(region):
    cy,cx=region.centroid_local
    cy/=region.image.shape[0]
    cx/=region.image.shape[1]
    perimeter=region.perimeter/region.image.size
    holes=count_holes(region)
    vlines = (np.sum(region.image,0) == region.image.shape[0]).sum() / region.image.shape[1]
    hlines = (np.sum(region.image,1) == region.image.shape[1]).sum() / region.image.shape[0]
    eccentricity=region.eccentricity
    h, w = region.image.shape
    aspect = min(h, w) / max(h, w)  
    vertikal_sum=vertik_symm(region)
    return np.array([region.area/region.image.size,cx,cy,perimeter,holes,vlines,hlines,eccentricity,aspect,vertikal_sum])
  
def classificator(region, templates):
    features = extractor(region)
    result = ""
    min_d = 10**16
    for symbol, t in templates.items():
        d = ((t - features)**2).sum() **0.5
        if d < min_d:
            result = symbol
            min_d = d
    return result

template = imread("C:/Users/anaer/OneDrive/Desktop/учеба/ИГУ/computer_vision/vector_recognition/alphabet-small.png")[:,:,:-1]
#print(template.shape)
template = template.sum(2)
binary = template != 765.

labeled = label(binary)
props = regionprops(labeled)
#print(type(props))
templates = {}
for region, symbol in zip (props,["8", "O",
                                  "A", "B", "1", "W",
                                  "X","*", "/","-" ]):
    templates[symbol] = extractor(region)

image = imread("C:/Users/anaer/OneDrive/Desktop/учеба/ИГУ/computer_vision/vector_recognition/alphabet.png")[:,:,:-1]
abinary = image.mean(2)>0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)
results = {}
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)
print(count_holes(aprops[1]))
#plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in results:
        results[symbol] = 0
    results[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print(results)
#print(templates)
#print(classificator(props[0],templates))
#plt.imshow(abinary)
#plt.show() 