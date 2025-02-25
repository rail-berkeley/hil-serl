from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_TRIANGLES)
    glVertex2f(-0.6, -0.6)
    glVertex2f(0.6, -0.6)
    glVertex2f(0.0, 0.6)
    glEnd()
    glFlush()

glutInit()
glutInitDisplayMode(GLUT_RGB)
glutInitWindowSize(400, 400)
glutCreateWindow("OpenGL Test")
glutDisplayFunc(draw)
glutMainLoop()
