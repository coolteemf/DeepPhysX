"""Intersection of two polygonal meshes"""
from vedo import *
import os

liver_file = os.getcwd() + '/liver.vtk'
liver = Mesh(liver_file).alpha(0.2)

vein_file = os.getcwd() + '/venacava.vtk'
vein = Mesh(vein_file).c("violet", 0.2)

contour = liver.intersectWith(vein).lineWidth(4).c('black')
boundary = liver.boundaries()
print(contour.points())

show(liver, vein, contour, boundary, __doc__, axes=7).close()
