import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path("C:/Users/anaer/OneDrive/Desktop/учеба/ИГУ/computer_vision/trajectory/motion/out")
files = sorted(path.iterdir(), key=lambda p: int(p.stem.split("_")[1]))
def area(labeled, label):
    return (labeled == label).sum()
def centroid(labeled, label):
    y, x = np.where(labeled == label)
    return np.mean(x), np.mean(y)
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
tracks = []
for f in files:
    mask = np.load(f)
    h, w = mask.shape
    lab = np.zeros_like(mask, int)
    label = 1
    for y in range(h):
        for x in range(w):
            if mask[y,x] and not lab[y,x]:
                s=[(y,x)]
                lab[y,x]=label
                while s:
                    cy,cx=s.pop()
                    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ny,nx=cy+dy,cx+dx
                        if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not lab[ny,nx]:
                            lab[ny,nx]=label
                            s.append((ny,nx))
                label+=1
    centers = [centroid(lab,i) for i in range(1,label) if area(lab,i) > 10]
    centers.sort()
    if not tracks:
        tracks = [[c] for c in centers]
        continue
    used = [0]*len(centers)
    for t in tracks:
        p = t[-1]
        best, idx = 9999, -1
        for j,c in enumerate(centers):
            if not used[j]:
                d = distance(p, c)
                if d < best:
                    best, idx = d, j
        if idx != -1 and best < 100:
            t.append(centers[idx])
            used[idx] = 1
    for j,c in enumerate(centers):
        if not used[j]:
            tracks.append([c])
for t in tracks:
    if len(t) > 5:
        x,y = zip(*t)
        plt.plot(x,y,'o-')
plt.gca().invert_yaxis()
plt.title('Траектории движения')
plt.show()