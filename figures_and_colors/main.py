import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
from skimage.color import rgb2hsv
from collections import Counter

image=imread("balls_and_rects.png")
hsv=rgb2hsv(image)
hue=hsv[:,:,0]  
value=hsv[:,:,2] 
mask = value> 0
labeled=label(mask)
regions=regionprops(labeled)
rectangle=[]
circle=[]

for prop in regions:
    ecs=prop.eccentricity
    cy,cx=prop.centroid
    color = round(hue[int(cy), int(cx)],2)
    if ecs<0.4:
        circle.append(color)
    else:
        rectangle.append(color)

count_c=Counter(circle)
count_r=Counter(rectangle)
print(len(regions))
print("Круги:")
for h, count in sorted(count_c.items()):
    print(f"{h:.6f} {count}")
print("Прямоугольники:")
for h, count in sorted(count_r.items()):
    print(f"{h:.6f} {count}")




