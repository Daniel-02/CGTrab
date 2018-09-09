import numpy as np
import cv2
from matplotlib import pyplot as plt

# Letter A
letterAvertsOg = [[-75, 50],
                [0, -100],
                [-75, 50],
                [-38, 50],
                [-25, 25],
                [75, 50],
                [37, 50],
                [25, 25],
                [-25, 0],
                [25, 0],
                [0, -50],
                ]
letterAedges = [[0, 1],
                [0, 3],
                [3, 4],
                [1, 5],
                [6, 5],
                [6, 7],
                [4, 7],
                [8, 9],
                [8, 10],
                [9, 10],
                ]
aDrawn = False
# mouse callback function
def draw_A_click(event,x,y,flags,param):
    global aDrawn
    if event == cv2.EVENT_LBUTTONDOWN:
        if not aDrawn:
            draw_A(x,y)
            aDrawn = True
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
# Draw letter A edges
def draw_A(x,y):
    global letterAvertsOg, letterAedges
    letterAverts = [[x + n[0], y + n[1]] for n in letterAvertsOg]
    cv2.line(img, tuple(letterAverts[letterAedges[0][0]]), tuple(letterAverts[letterAedges[0][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[1][0]]), tuple(letterAverts[letterAedges[1][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[2][0]]), tuple(letterAverts[letterAedges[2][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[3][0]]), tuple(letterAverts[letterAedges[3][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[4][0]]), tuple(letterAverts[letterAedges[4][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[5][0]]), tuple(letterAverts[letterAedges[5][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[6][0]]), tuple(letterAverts[letterAedges[6][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[7][0]]), tuple(letterAverts[letterAedges[7][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[8][0]]), tuple(letterAverts[letterAedges[8][1]]), (255, 0, 0), 4)
    cv2.line(img, tuple(letterAverts[letterAedges[9][0]]), tuple(letterAverts[letterAedges[9][1]]), (255, 0, 0), 4)

# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
# Create a black image
img = np.zeros((600, 900, 3), np.uint8)
cv2.namedWindow('ESC para sair')
cv2.setMouseCallback('ESC para sair',draw_A_click)
while (1):
    cv2.imshow('ESC para sair', img)
    # cv2.line(img, (0,0), (x, x), (255, 0, 0))
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
