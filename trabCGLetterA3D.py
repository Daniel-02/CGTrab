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
                [2, 0],
                [3, 2],
                [1, 4],
                [4, 5],
                [5, 6],
                [6, 3],
                [8, 7],
                [7, 9],
                [9, 8],  # 9
                [11, 10],  # Invertido
                [10, 12],  # Invertido
                [12, 13],  # Invertido
                [14, 11],  # Invertido
                [15, 14],  # Invertido
                [16, 15],  # Invertido
                [13, 16],  # Invertido
                [17, 18],
                [19, 17],
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
                [0, 2],    # 30
                [12, 10],  # 31
                [10, 0],   # 32
                [10, 11],  # 33
                [11, 1],   # 34
                [1, 0],    # 35
                [11, 14],  # 36
                [14, 4],   # 37
                [4, 1],    # 38
                [14, 15],  # 39
                [15, 5],   # 40
                [5, 4],    # 41
                [15, 16],  # 42
                [16, 6],   # 43
                [6, 5],    # 44
                [16, 13],  # 45
                [13, 3],   # 46
                [3, 6],    # 47
                [2, 3],    # 48
                [13, 12],  # 49
                [12, 2],   # 50
                [17, 7],   # 51
                [19, 9],   # 52
                [18, 8],   # 53
                ]

letterAFaces = [[0, 3, 4, 5, 6, 2, 1],
                [11, 12, 16, 15, 14, 13, 10],
                [30, 22, 31, 32],  # Ok
                [20, 33, 34, 35],  # Ok
                [21, 36, 37, 38],  # Ok
                [24, 39, 40, 41],  # Ok
                [25, 42, 43, 44],  # OK
                [26, 45, 46, 47],  # Ok
                [48, 23, 49, 50],  # Ok
                [27, 17, 53, 7],   # Ok
                [8, 29, 18, 51],
                [28, 19, 52, 9],
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


# mouse callback function
# def draw_a_click(event, x, y, flags, param):
#     global aDrawn
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if not aDrawn:
#             draw_a(x, y)
#             aDrawn = True

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
    for letterAEdge in letterAEdges:
        cv2.line(img, tuple(coord_to_draw(verts[letterAEdge[0]])),
                 tuple(coord_to_draw(verts[letterAEdge[1]])), (255, 0, 0), 1)

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
        # paint_face(face, verts, color*cos)
        i += 1
    coss.sort()
    for cos in coss:
        if cos[0] == 0:
            paint_face(cos[1], verts, (0, 0, 0))
        else:
            paint_face(cos[1], verts, color*cos[0])


def paint_face(face, a_verts, color):
    verts = []
    for edge in edges_to_draw([face]):
        for vert in letterAEdges[edge]:
            if coord_to_draw(a_verts[vert]) not in verts:
                verts.append(coord_to_draw(a_verts[vert]))
    cv2.fillPoly(img, np.array([verts]), color)
    if face == [0, 3, 4, 5, 6, 2, 1]:
        cv2.fillPoly(img, np.array([[coord_to_draw(a_verts[7]),
                                     coord_to_draw(a_verts[8]),
                                     coord_to_draw(a_verts[9])]]), (0, 0, 0))
    if face == [11, 12, 16, 15, 14, 13, 10]:
        cv2.fillPoly(img, np.array([[coord_to_draw(a_verts[17]),
                                     coord_to_draw(a_verts[18]),
                                     coord_to_draw(a_verts[19])]]), (0, 0, 0))


# Create a black image
img = np.zeros((yDisplay, xDisplay, 3), np.uint8)  # (Y, X) do display
# cv2.namedWindow('ESC para sair')
# cv2.setMouseCallback('ESC para sair', draw_a_click)
frameCount = 0
wireframe = True
observer = False
paint = False


while frameCount < totalFrames:
    if wireframe:
        # calculate_a(xMovInit - ((xMovInit - xMovEnd) / totalFrames) * frameCount,
        #             yMovInit - ((yMovInit - yMovEnd) / totalFrames) * frameCount, frameCount, 2*np.pi/3)
        calculate_a(50, 50, 0, 2*np.pi/3)
    if observer:
        # calculate_a_observer(xMovInit - ((xMovInit - xMovEnd) / totalFrames) * frameCount,
        #             yMovInit - ((yMovInit - yMovEnd) / totalFrames) * frameCount, frameCount, 2*np.pi/3)
        calculate_a_observer(50, 50, 0, 2*np.pi/3)
    if paint:
        # calculate_a_to_paint(xMovInit - ((xMovInit - xMovEnd) / totalFrames) * frameCount,
        #             yMovInit - ((yMovInit - yMovEnd) / totalFrames) * frameCount, frameCount, 2*np.pi/3)
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
