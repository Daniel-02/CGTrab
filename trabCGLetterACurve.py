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
                  [-8, 0, 0, 1],   # Ponto de baixo-meio esquerdo 2
                  [-6, 2, 0, 1],   # Ponto de cima-meio esquerdo 3
                  [0, 0, 0, 1],    # Ponto de baixo direito 4
                  [-2, 0, 0, 1],   # Ponto de baixo-meio direito 5
                  [-4, 2, 0, 1],   # Ponto de cima-meio direito 6
                  [-10, 0, 2, 1],  # 7
                  [-5, 19, 2, 1],
                  [-8, 0, 2, 1],
                  [-6, 2, 2, 1],
                  [0, 0, 2, 1],
                  [-2, 0, 2, 1],
                  [-4, 2, 2, 1],
#       Pontos intermediarios da curva de Bezier
                  [-8, 1, 0, 1],
                  [-7, 2, 0, 1],
                  [-8, 1, 2, 1],
                  [-7, 2, 2, 1],
                  [-2, 1, 0, 1],
                  [-3, 2, 0, 1],
                  [-2, 1, 2, 1],
                  [-3, 2, 2, 1],
                  ]

letterAEdges = [
                [2, 0],
                [3, 2],
                [4, 5],
                [5, 6],
                [6, 3],
                [7, 9],
                [9, 10],
                [12, 11],
                [13, 12],
                [10, 13],
                [0, 7],
                [1, 8],
                [2, 9],
                [3, 10],
                [4, 11],
                [5, 12],
                [6, 13],
                ]

letterACurves = [
                 [0, 14, 15, 1],
                 [7, 16, 17, 8],
                 [4, 18, 19, 1],
                 [11, 20, 21, 8]
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
quatAxis = np.array([1, 1, 1])
rotAngle = 2*np.pi


def convert_x_universe_to_x_display(x_u):
    global xUniverse, xDisplay
    return int(round(x_u * xDisplay / xUniverse))


def convert_y_universe_to_y_display(y_u):
    global yUniverse, yDisplay
    return int(round(y_u * ((-yDisplay) / yUniverse) + yDisplay))


def coord_to_draw(coord):
    return [convert_x_universe_to_x_display(coord[0] / coord[3]), convert_y_universe_to_y_display(coord[1] / coord[3])]


def curve_to_display(curve):
    return [coord_to_draw(point) for point in curve]


def trans_rot_points(points, x, y, frameCount):
    transMatrix = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [x, y, 0, 1]]

    harmonic_value = [point[-1] for point in points]
    tempPoints = [quat_rot(np.array(point[:-1]), (rotAngle/totalFrames)*frameCount) for point in points]
    tempPoints = [np.append(point, harmonic_value[index]) for index, point in enumerate(tempPoints)]
    tempPoints = [np.matmul(n, transMatrix) for n in tempPoints]
    return tempPoints


def proj_points(points, ang):
    projMatrix = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [np.cos(ang), np.sin(ang), 0, 0],
                  [0, 0, 0, 1]]
    return np.matmul(points, projMatrix).tolist()




def calculate_a(x, y, frameCount, ang):
    letter_a_verts = trans_rot_points(letterAvertsOg, x, y, frameCount)
    letter_a_verts = proj_points(letter_a_verts, ang)
    draw_a_edges(letter_a_verts, letterAEdges)


def draw_a_edges(verts, edges):
    for letterAEdge in edges:
        cv2.line(img, tuple(coord_to_draw(verts[letterAEdge[0]])),
                 tuple(coord_to_draw(verts[letterAEdge[1]])), (255, 0, 0), 1)


def draw_a_curves(curves):
    for curve in curves:
        cv2.polylines(img, [np.array(curve_to_display(curve), np.int32)], False, (255, 0, 0), 1)


def draw_A_curves(x, y, frameCount, ang):
    letter_a_verts = trans_rot_points(letterAvertsOg, x, y, frameCount)
    curves = []
    for curveVerts in letterACurves:
        curves.append(proj_points(calc_bezier(letter_a_verts[curveVerts[0]], letter_a_verts[curveVerts[1]],
                                  letter_a_verts[curveVerts[2]], letter_a_verts[curveVerts[3]]), ang))
    draw_a_curves(curves)


def calc_bezier(bStart, bInt1, bInt2, bEnd):
    curve = []
    for i in range(0, 1000):
        u = i/1000
        bPointX = ((1-u)**3)*bStart[0] + (3*u*(1-u)**2)*bInt1[0] + (3*(1-u)*u**2)*bInt2[0] + (u**3)*bEnd[0]
        bPointY = ((1-u)**3)*bStart[1] + (3*u*(1-u)**2)*bInt1[1] + (3*(1-u)*u**2)*bInt2[1] + (u**3)*bEnd[1]
        bPointZ = ((1-u)**3)*bStart[2] + (3*u*(1-u)**2)*bInt1[2] + (3*(1-u)*u**2)*bInt2[2] + (u**3)*bEnd[2]
        curve.append([bPointX, bPointY, bPointZ, 1])
    return curve


def quat_rot(point, angle):
    s = np.cos(angle/2)
    v = np.sin(angle/2)*quatAxis
    return ((s**2)*point) - (np.dot(v, v)*point) + (2*np.dot(v, point)*v) + (2*s*np.cross(v, point))


# Create a black image
img = np.zeros((yDisplay, xDisplay, 3), np.uint8)  # (Y, X) do display
frameCount = 0

while frameCount < totalFrames:
    calculate_a(50, 50, frameCount, 2*np.pi/3)
    draw_A_curves(50, 50, frameCount, 2*np.pi/3)
    sleep(0.01)
    cv2.imshow('ESC para sair', img)
    cv2.rectangle(img, (0, 0), (xDisplay, yDisplay), (0, 0, 0), -1)
    frameCount += 1
    if frameCount == totalFrames:
        frameCount = 1
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
