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
                  [-10, 0, 2, 1],  # 10
                  [-5, 19, 2, 1],
                  [-7, 0, 2, 1],
                  [-6, 3, 2, 1],
                  [0, 0, 2, 1],
                  [-3, 0, 2, 1],
                  [-4, 3, 2, 1],
                  ]

letterAEdges = [[0, 1],
                [2, 0],
                [3, 2],
                [1, 4],
                [4, 5],
                [5, 6],
                [6, 3],
                [8, 7],
                [7, 9],
                [9, 10],
                [11, 8],
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
                [0, 2],    # 30
                [9, 7],  # 31
                [7, 0],   # 32
                [7, 8],  # 33
                [8, 1],   # 34
                [1, 0],    # 35
                [8, 11],  # 36
                [11, 4],   # 37
                [4, 1],    # 38
                [11, 12],  # 39
                [12, 5],   # 40
                [5, 4],    # 41
                [12, 13],  # 42
                [13, 6],   # 43
                [6, 5],    # 44
                [13, 10],  # 45
                [10, 3],   # 46
                [3, 6],    # 47
                [2, 3],    # 48
                [10, 9],  # 49
                [9, 2],   # 50
                ]

letterAFaces = [[0, 3, 4, 5, 6, 2, 1],
                [8, 9, 13, 12, 11, 10, 7],
                [21, 16, 22, 23],  # Ok
                [14, 24, 25, 26],  # Ok
                [15, 27, 28, 29],  # Ok
                [18, 30, 31, 32],  # Ok
                [19, 33, 34, 35],  # OK
                [20, 36, 37, 38],  # Ok
                [39, 17, 40, 41],  # Ok
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
observerVertice = [200, 200, 100]
paintedVertice = [200, 200, 100]

def convert_x_universe_to_x_display(x_u):
    global xUniverse, xDisplay
    return int(round(x_u * xDisplay / xUniverse))


def convert_y_universe_to_y_display(y_u):
    global yUniverse, yDisplay
    return int(round(y_u * ((-yDisplay) / yUniverse) + yDisplay))


def coord_to_draw(coord):
    return [convert_x_universe_to_x_display(coord[0] / coord[3]), convert_y_universe_to_y_display(coord[1] / coord[3])]


def normal_vectors(aVerts):
    normals = []
    for face in letterAFaces:
        vert0 = np.array(aVerts[letterAEdges[face[0]][0]][:-1])
        vert1 = np.array(aVerts[letterAEdges[face[0]][1]][:-1])
        vert2 = np.array(aVerts[letterAEdges[face[1]][1]][:-1])
        normals.append(np.cross((vert1 - vert0), (vert2 - vert0)))
    return normals


def visible_faces(normals, aVerts):
    visibleFaces = []
    obsVert = np.array(observerVertice)
    i = 0
    for face in letterAFaces:
        vert0 = np.array(aVerts[letterAEdges[face[0]][0]][:-1])
        if np.dot((vert0 - obsVert), normals[i]) >= 0:
            visibleFaces.append(face)
        i += 1
    return visibleFaces


def edges_to_draw(faces):
    edges = []
    for face in faces:
        for edge in face:
            if edge not in edges:
                edges.append(edge)
    return edges


def calculate_a_observer(x, y, frameCount, ang):
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
    normals = normal_vectors(letter_a_verts)
    faces = visible_faces(normals, letter_a_verts)
    letter_a_verts = np.matmul(letter_a_verts, projMatrix).tolist()
    draw_a_edges(letter_a_verts, edges_to_draw(faces))


def calculate_a_to_paint(x, y, frameCount, ang):
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
    normals = normal_vectors(letter_a_verts)
    letter_a_verts = np.matmul(letter_a_verts, projMatrix).tolist()
    paint_a_faces(letter_a_verts, normals)


def calculate_a(x, y, frameCount, ang):
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
    draw_a_edges(letter_a_verts, edges_to_draw(letterAFaces))


# Draw letter A edges
def draw_a(verts):
    for letterAEdge in letterAEdges:
        cv2.line(img, tuple(coord_to_draw(verts[letterAEdge[0]])), tuple(coord_to_draw(verts[letterAEdge[1]])), (255, 0, 0), 1)


def draw_a_edges(verts, edges):
    for edge in edges:
        letterAEdge = letterAEdges[edge]
        cv2.line(img, tuple(coord_to_draw(verts[letterAEdge[0]])),
                 tuple(coord_to_draw(verts[letterAEdge[1]])), (255, 0, 0), 1)


def paint_a_faces(verts, normals):
    color = np.array([30, 135, 255])
    i = 0
    colV = np.array(paintedVertice)
    coss = []
    for face in letterAFaces:
        norm = np.array(normals[i])
        # sin = np.linalg.norm(np.cross(obsV, norm))/(np.linalg.norm(obsV)*np.linalg.norm(norm))
        cos = (np.dot(colV, norm)/(np.linalg.norm(colV)*np.linalg.norm(norm)))
        if cos < 0:
            cos = 0
        coss.append((cos, face))
        i += 1
    coss.sort()
    for cos in coss:
        if cos[0] == 0:
            paint_face(cos[1], verts, (0, 0, 0))
        else:
            paint_face(cos[1], verts, color*cos[0])

def parabola(xStart, xEnd):
    curve = []
    for x in range(xStart,xEnd):
        y = 0.0425*x*x - 6.25*x + 258
        curve.append([2*x, 2*y])
    return curve

def paint_face(face, a_verts, color):
    verts = []
    for edge in edges_to_draw([face]):
        for vert in letterAEdges[edge]:
            if coord_to_draw(a_verts[vert]) not in verts:
                verts.append(coord_to_draw(a_verts[vert]))
    cv2.fillPoly(img, np.array([verts]), (0+color[0], 0+color[1], 50+color[2]))


# Create a black image
img = np.zeros((yDisplay, xDisplay, 3), np.uint8)  # (Y, X) do display
frameCount = 0
wireframe = True
observer = False
paint = False
curve = parabola(20, 50)
curve = np.array(curve, np.int32)

while frameCount < totalFrames:
    cv2.polylines(img, [curve], False, (255, 255, 255), 2)
    if wireframe:
        calculate_a(50, 50, 0, 2*np.pi/3)
    if observer:
        calculate_a_observer(50, 50, 0, 2*np.pi/3)
    if paint:
        calculate_a_to_paint(50, 50, 0, 2*np.pi/3)
    sleep(0.005)
    cv2.imshow('ESC para sair', img)
    cv2.rectangle(img, (0, 0), (xDisplay, yDisplay), (0, 0, 0), -1)
    frameCount += 1
    if frameCount == totalFrames:
        frameCount = 1
    if cv2.waitKey(5) & 0xFF == 27:
        break
    if cv2.waitKey(5) == ord('1'):
        wireframe = True
        observer = False
        paint = False
        frameCount = 1
    if cv2.waitKey(5) == ord('2'):
        observer = True
        wireframe = False
        paint = False
        frameCount = 1
    if cv2.waitKey(5) == ord('3'):
        paint = True
        observer = False
        wireframe  = False
        frameCount = 1

cv2.destroyAllWindows()
