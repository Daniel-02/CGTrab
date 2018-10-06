import numpy as np
import cv2
from time import sleep
from screeninfo import get_monitors

# Letter A
# letterAvertsT = [[-75, 150],
#                  [0, 0],
#                  [-75, 150],
#                  [-38, 150],
#                  [-25, 125],
#                  [75, 150],
#                  [37, 150],
#                  [25, 125],
#                  [-25, 100],
#                  [25, 100],
#                  [0, 50],
#                  ]

# letterAvertsOg = [[-5, 0, 1],
#                   [0, 19, 1],
#                   [-5, 0, 1],
#                   [-2, 0, 1],
#                   [-1, 3, 1],
#                   [5, 0, 1],
#                   [2, 0, 1],
#                   [1, 3, 1],
#                   [-1, 6, 1],
#                   [1, 6, 1],
#                   [0, 12, 1],
#                   ]
letterAvertsOg = [[-10, 0, 1],
                  [-5, 19, 1],
                  [-10, 0, 1],
                  [-7, 0, 1],
                  [-6, 3, 1],
                  [0, 0, 1],
                  [-3, 0, 1],
                  [-4, 3, 1],
                  [-6, 6, 1],
                  [-4, 6, 1],
                  [-5, 12, 1],
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
xDisplay = 1200
yDisplay = 900
xUniverse = 100
yUniverse = 100
xMovInit = 100
yMovInit = 42
xMovEnd = 0
yMovEnd = 100
totalFrames = 100


def convert_x_universe_to_x_display(x_u):
    global xUniverse, xDisplay
    return int(round(x_u * xDisplay / xUniverse))


def convert_y_universe_to_y_display(y_u):
    global yUniverse, yDisplay
    return int(round(y_u * ((-yDisplay) / yUniverse) + yDisplay))


def coord_to_draw(coord):
    return [convert_x_universe_to_x_display(coord[0] / coord[2]), convert_y_universe_to_y_display(coord[1] / coord[2])]


# mouse callback function
# def draw_a_click(event, x, y, flags, param):
#     global aDrawn
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not aDrawn:
#             draw_a(x, y)
#             aDrawn = True


def calculate_a(x, y, frameCount):
    # global letterAvertsT, letterAedges
    transMatrix = [[1, 0, 0],
                   [0, 1, 0],
                   [x, y, 1]]
    rotMatrix = [
        [np.cos((np.pi / (2 * totalFrames)) * frameCount), np.sin((np.pi / (2 * totalFrames)) * frameCount), 0],
        [-np.sin((np.pi / (2 * totalFrames)) * frameCount), np.cos((np.pi / (2 * totalFrames)) * frameCount), 0],
        [0, 0, 1]]
    letter_a_verts = [np.matmul(n, np.matmul(rotMatrix, transMatrix)).tolist() for n in letterAvertsOg]
    draw_a(letter_a_verts)


# Draw letter A edges
def draw_a(verts):
    for letterAedge in letterAedges:
        cv2.line(img, tuple(coord_to_draw(verts[letterAedge[0]])),
                 tuple(coord_to_draw(verts[letterAedge[1]])), (255, 0, 0), 4)


# Get screen display
for m in get_monitors():
    print(str(m))
# Create a black image
img = np.zeros((yDisplay, xDisplay, 3), np.uint8)  # (Y, X) do display
# cv2.namedWindow('ESC para sair')
# cv2.setMouseCallback('ESC para sair', draw_a_click)
frameCount = 0
while frameCount < totalFrames:
    calculate_a(xMovInit - ((xMovInit - xMovEnd) / totalFrames) * frameCount,
                yMovInit - ((yMovInit - yMovEnd) / totalFrames) * frameCount, frameCount)
    sleep(0.02)
    cv2.imshow('ESC para sair', img)
    cv2.rectangle(img, (0, 0), (xDisplay, yDisplay), (0, 0, 0), -1)
    frameCount += 1
    if frameCount == totalFrames:
        frameCount = 1
    if cv2.waitKey(20) & 0xFF == 27:
        break


cv2.destroyAllWindows()
