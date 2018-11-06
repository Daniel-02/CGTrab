import numpy as np
import cv2
from time import sleep

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
letterAvertsOg = [[-10, 0, 0, 1],  # Ponto de baixo esquerdo 0
                  [-5, 19, 0, 1],  # Ponto da ponta de cima 1
                  [-7, 0, 0, 1],   # Ponto de baixo-meio esquerdo 2
                  [-6, 3, 0, 1],   # Ponto de cima-meio esquerdo 3
                  [0, 0, 0, 1],    # Ponto de baixo direito 4
                  [-3, 0, 0, 1],   # Ponto de baixo-meio direito 5
                  [-4, 3, 0, 1],   # Ponto de cima-meio direito 6
                  [-6, 6, 0, 1],   # Ponto do meio esquerdo 7
                  [-4, 6, 0, 1],   # Ponto do meio direito 8
                  [-5, 12, 0, 1],  # Ponto da ponta do meio 9
                  [-10, 0, 2, 1],  # 10
                  [-5, 19, 2, 1],
                  [-7, 0, 2, 1],
                  [-6, 3, 2, 1],
                  [0, 0, 2, 1],
                  [-3, 0, 2, 1],
                  [-4, 3, 2, 1],
                  [-6, 6, 2, 1],
                  [-4, 6, 2, 1],
                  [-5, 12, 2, 1],
                  ]

letterAEdges = [[0, 1],
                [0, 2],
                [2, 3],
                [1, 4],
                [5, 4],
                [5, 6],
                [3, 6],
                [7, 8],
                [7, 9],
                [8, 9],  # 9
                [10, 11],
                [10, 12],
                [12, 13],
                [11, 14],
                [15, 14],
                [15, 16],
                [13, 16],
                [17, 18],
                [17, 19],
                [18, 19],  # 19
                [0, 10],
                [1, 11],
                [2, 12],
                [3, 13],
                [4, 14],
                [5, 15],
                [6, 16],
                [7, 17],
                [8, 18],
                [9, 19],
                ]

letterAFaces = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [0, 20, 10, 21],
                [1, 20, 11, 22],
                [2, 22, 12, 23],
                [6, 23, 16, 26],
                [5, 25, 15, 26],
                [4, 25, 14, 24],
                [3, 24, 13, 21],
                [7, 27, 17, 28],
                [8, 27, 18, 29],
                [9, 28, 19, 29],
                ]

xDisplay = 1200
yDisplay = 900
xUniverse = 100
yUniverse = 100
xMovInit = 100
yMovInit = 42
xMovEnd = 0
yMovEnd = 100
totalFrames = 75


def convert_x_universe_to_x_display(x_u):
    global xUniverse, xDisplay
    return int(round(x_u * xDisplay / xUniverse))


def convert_y_universe_to_y_display(y_u):
    global yUniverse, yDisplay
    return int(round(y_u * ((-yDisplay) / yUniverse) + yDisplay))


def coord_to_draw(coord):
    return [convert_x_universe_to_x_display(coord[0] / coord[3]), convert_y_universe_to_y_display(coord[1] / coord[3])]


# mouse callback function
# def draw_a_click(event, x, y, flags, param):
#     global aDrawn
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not aDrawn:
#             draw_a(x, y)
#             aDrawn = True


def calculate_a(x, y, frameCount, ang):
    # global letterAvertsT, letterAEdges
    transMatrix = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [x, y, 0, 1]]
    rotMatrix = [
        [np.cos((np.pi / (2 * totalFrames)) * frameCount), np.sin((np.pi / (2 * totalFrames)) * frameCount), 0, 0],
        [-np.sin((np.pi / (2 * totalFrames)) * frameCount), np.cos((np.pi / (2 * totalFrames)) * frameCount), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    projMatrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [np.cos(ang), np.sin(ang), 0, 0],
                  [0, 0, 0, 1]]
    letter_a_verts = [np.matmul(n, np.matmul(rotMatrix, transMatrix)) for n in letterAvertsOg]
    letter_a_verts = np.matmul(letter_a_verts, projMatrix).tolist()
    draw_a(letter_a_verts)


# Draw letter A edges
def draw_a(verts):
    for letterAedge in letterAEdges:
        cv2.line(img, tuple(coord_to_draw(verts[letterAedge[0]])),
                 tuple(coord_to_draw(verts[letterAedge[1]])), (255, 0, 0), 1)


# Create a black image
img = np.zeros((yDisplay, xDisplay, 3), np.uint8)  # (Y, X) do display
# cv2.namedWindow('ESC para sair')
# cv2.setMouseCallback('ESC para sair', draw_a_click)
frameCount = 0
while frameCount < totalFrames:
    calculate_a(xMovInit - ((xMovInit - xMovEnd) / totalFrames) * frameCount,
                yMovInit - ((yMovInit - yMovEnd) / totalFrames) * frameCount, frameCount, 2*np.pi/3)
    # calculate_a(50, 50, 2*np.pi/3)
    sleep(0.015)
    cv2.imshow('ESC para sair', img)
    cv2.rectangle(img, (0, 0), (xDisplay, yDisplay), (0, 0, 0), -1)
    frameCount += 1
    if frameCount == totalFrames:
        frameCount = 1
    if cv2.waitKey(20) & 0xFF == 27:
        break


cv2.destroyAllWindows()
