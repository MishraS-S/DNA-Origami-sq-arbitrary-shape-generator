# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:10:08 2020

@author: Shubham
"""

#Trimesh stores faces as triplets of vertex indices, edges as duplets of vertex indices
#.facets object only has non-triangular "faces", each facet is denoted by component face indices

#3. Modifying vertices (rotating faces to XY plane) in-place doesn't work, because flattening one face
#applies a non-affine transformation to all adjacent faces - therefore probably need to make my own
#polygon object preserving vertex, edge and face information (make sure no difference between face and facet
#in this new object)


import trimesh

from scipy.spatial import distance
import numpy as np
from copy import deepcopy

#mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
              #         faces=[[0, 1, 2]],
         #              process=False)

poly = trimesh.load(file_obj=r"C:\Users\Shubham\Downloads\tinker1.obj", file_type='obj')

# poly

# poly.show()

# poly.faces

# poly.facets

# poly.facets_boundary

# poly.vertices

# newface = trimesh.Trimesh(vertices=poly.vertices[[0,1,2,3]], faces=[[0,1,2,3]], process=False)
# newface = poly.facets[0]

# newface.show()



#Rotating facet to XY plane
facetindex = 2
zax = np.array([0,0,1]) #Z-axis
ang = np.arccos(np.dot(poly.facets_normal[facetindex], zax)) #angle between Z-axis and normal of facet
rotax = np.cross(poly.facets_normal[facetindex], zax) #Vector of rotation axis
rotmat = trimesh.transformations.rotation_matrix(float(ang), rotax, point=[0,0,0]) #rotation matrix
myvec = np.append(poly.facets_normal[facetindex],0) #Add a zero at the end of normal of facet to be rotated so that we can multiply with rotation matrix (which is 4x4)
xynorm = np.matmul(rotmat,myvec) #Normal Vector of facet after rotation (should be Z-axis)

ogfacetvertsindices = []
for face in poly.facets[facetindex]:
    for vert in poly.faces[face]:
        ogfacetvertsindices.append(vert)
ogfacetverts = []
ogfacetvertsindices = list(set(ogfacetvertsindices))
for vert in ogfacetvertsindices:
    ogfacetverts.append(poly.vertices[vert])

newverts = []
for vert in ogfacetverts:
    vert = np.append(vert,0) #Add zero to vertex vector for multiplication purpose
    rotvert = np.matmul(rotmat,vert)
    newverts.append(rotvert)


newverts = np.asarray(newverts)
newverts = np.delete(newverts, 3, 1) #delete last column
for ind, ob in enumerate(ogfacetvertsindices):
    poly.vertices[ob] =  newverts[ind]


faceindex = []
for i in range(len(newverts)):
    faceindex.append(int(i))
print(faceindex)
newrotface = trimesh.Trimesh(vertices=newverts, faces=[faceindex], process=False)
print(newrotface.vertices)
print(newrotface.facets_normal[0])
print(xynorm)



# newface.vertices

# newrotface.vertices[0]

# distance.euclidean(newrotface.vertices[1],newrotface.vertices[2])

# distance.euclidean(newface.vertices[1],newface.vertices[2])