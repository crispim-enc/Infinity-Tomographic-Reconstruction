from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtOpenGL import *
import math
import time

FPS_TARGET = 50

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.widget = glWidget(self)
        self.button = QtWidgets.QPushButton('Test', self)
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        mainLayout.addWidget(self.button)
        self.setLayout(mainLayout)




class glWidget(QGLWidget):

    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(640, 480)



    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(-2.5, 0.5, -6.0)
        glColor3f( 1.0, 1.5, 0.0 )
        glPolygonMode(GL_FRONT, GL_FILL)
        # glBegin(GL_TRIANGLES)
        # glVertex3f(2.0,-1.2,0.0)
        # glVertex3f(2.6,0.0,0.0)
        # glVertex3f(2.9,-1.2,0.0)
        #
        # glEnd()
        # glOrtho(-10, 110, -10, 70, -1, 1)

        # glTranslatef(0, 0, -100)        #move back
        # glRotatef(-20, 1, 0, 0)             #orbit higher

        glFlush()

    def initializeGL(self):
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0,1.33,0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)



    def axis(self,i):
        glBegin(GL_LINES)

        # x = red
        # y = green
        # z = blue

        glColor3f(1, 0, 0)
        glVertex3fv((0, 0, 0))
        glVertex3fv((1, 0, 0))

        glColor3f(0, 1, 0)
        glVertex3fv((0, 0, 0))
        glVertex3fv((0, 1, 0))

        glColor3f(0, 0, 1)
        glVertex3fv((0, 0, 0))
        glVertex3fv((0, 0, 1))

        glEnd()

    def quad(self,points, color):
        glBegin(GL_QUADS)
        glColor3f(*color)
        for p in points:
            glVertex3fv(p)
        glEnd()

    def cquad(self,point, size, color):
        glBegin(GL_QUADS)
        glColor3f(*color)
        x, y, z = point
        s = size / 2.0
        glVertex3fv((x - s, y - s, z))
        glVertex3fv((x + s, y - s, z))
        glVertex3fv((x + s, y + s, z))
        glVertex3fv((x - s, y + s, z))
        glEnd()

    def tick(self,i):
        # glRotatef(1, 0, 0, 1)
        # glTranslatef(0, 0, 1)

        # Draw Axis
        self.axis(i)

        # Draw sinewave
        for x in range(200):
            x = x / 2.0
            y = math.sin(math.radians(x + i) * 10) * 30 + 30
            self.cquad((x, y, 0), 1, (y / 60.0, 0, x / 100.0))

if __name__ == '__main__':
    app = QtWidgets.QApplication(['Yo'])

    window = MainWindow()
    window.show()
    app.exec_()
    nt = int(time.time() * 1000)

    for i in range(2 ** 63):
        nt += 1000 // FPS_TARGET

        # check for quit'n events
        # event = pygame.event.poll()
        # if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        #     break

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # self.tick(i)

        # pygame.display.flip()

        ct = int(time.time() * 1000)
        # pygame.time.wait(max(1, nt - ct))

        if i % FPS_TARGET == 0:
            print(nt - ct)

