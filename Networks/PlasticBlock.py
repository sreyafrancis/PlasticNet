from math import radians
import random

import numpy as np
from itertools import product
from mathutils import Vector
from mathutils import Matrix
from mathutils import geometry
# from collections import Counter

"""
First : this data structure uses arrays to keep vertices' id
        What I thought was a better idea than a linked list turns out to be a terrible idea
        It should be redone that way....  
"""



class PlasticBlock:

    class Point:
        def __init__(self, p_id, x, y, z):
            self.p_id = p_id
            self.coord = [x, y, z]

    def add_vertex(self, x, y, z):
        p_id = self.vertices.__len__()
        self.vertices.append(Vector([x, y, z]))
        p = PlasticBlock.Point(p_id, x, y, z)

        return p

    def __init__(self, parameters=None,
                 eccentricity: int = random.randint(0, 4),
                 rot_ecc: int = random.randint(0, 3),
                 rot_mag: float = np.random.beta(2, 5),
                 shear_ecc: int = random.randint(0, 4),
                 shear_mag = random.randint(0, 3),
                 smoothness=6):

        if parameters is None :
            parameters = self.random_sample(eccentricity, rot_ecc, rot_mag, shear_ecc, shear_mag)

        # list tru all radius of curvature parameters and apply cap
        for i in range(0, len(parameters), 2):
            parameters[i] = min(parameters[i], 1)
            if parameters[i] > .95:
                parameters[i] = 1
            if parameters[i] < .05:
                parameters[i] = .05

        self.source = parameters
        self.params = parameters[:40].reshape((-1, 2))
        self.affine = parameters[40:]
        self.sub = max(int(smoothness), 2)
        self.vertices = []
        self.edge_list = []
        self.face_list = []
        # control points
        #   array of 8 control points
        self.control_points = []
        for p_id in range(8):
            p = self.add_vertex((((p_id >> 2) & 1) * -2) + 1, ((((p_id & -6) >> 1) & 1) * -2) + 1, (((p_id & -7) & 1) * -2) + 1)
            self.control_points.append(p)

        # array of 12 meta edges
        self.m_edges = np.empty((3, 4), dtype=object)
        # again edges' id are one-hot encoded related to control point's code
        # it's simpler to list them manually, but a bitwise function can do it
        # i is the bit position to be removed, index is read from the 2 remaining bits
        idx = [[4, 0], [5, 1], [6, 2], [7, 3]]
        idy = [[2, 0], [3, 1], [6, 4], [7, 5]]
        idz = [[1, 0], [3, 2], [5, 4], [7, 6]]
        for i, listi in enumerate([idx, idy, idz]):
            for j, index in enumerate(listi):
                self.m_edges[i][j] = PlasticBlock.Edge(self.control_points[index[0]], self.control_points[index[1]])
                p = self.add_vertex(self.m_edges[i][j].coord[0], self.m_edges[i][j].coord[1], self.m_edges[i][j].coord[2])
                self.m_edges[i][j].point = p
                self.m_edges[i][j].p_id = p.p_id
                self.m_edges[i][j].curve = self.params[p.p_id]

        # complete dual graph with faces
        self.faces = []
        # x edges :
        self.add_face(self.m_edges[1][0], self.m_edges[1][1], self.m_edges[2][0], self.m_edges[2][1])
        self.add_face(self.m_edges[1][2], self.m_edges[1][3], self.m_edges[2][2], self.m_edges[2][3])
        # y edges:
        self.add_face(self.m_edges[0][0], self.m_edges[0][1], self.m_edges[2][0], self.m_edges[2][2])
        self.add_face(self.m_edges[0][2], self.m_edges[0][3], self.m_edges[2][1], self.m_edges[2][3])
        # z edges:
        self.add_face(self.m_edges[0][0], self.m_edges[0][2], self.m_edges[1][0], self.m_edges[1][2])
        self.add_face(self.m_edges[0][1], self.m_edges[0][3], self.m_edges[1][1], self.m_edges[1][3])

        for c, point in enumerate(self.control_points):
            ex = self.m_edges[0][idx.index([x for x in idx if c in x][0])]
            ey = self.m_edges[1][idy.index([x for x in idy if c in x][0])]
            ez = self.m_edges[2][idz.index([x for x in idz if c in x][0])]
            self.control_points[c] = PlasticBlock.CtrPoint(self, c, point, ex, ey, ez, self.params[point.p_id], self.sub)

        self.get_edges()
        self.get_faces()

        for c in self.control_points:
            self.deform_sphere(c)

        for e in self.m_edges.reshape(12):
            self.deform_cylinder(e)

        self.apply_affine(self.affine)
        # Counter can check for duplicates
        # print(Counter(tuple(item) for item in self.edge_list))
        a = 1

    class FacePoint:
        # Create a face with 4 co-planar edges
        def __init__(self, e1, e2, e3, e4):
            self.coord = (np.equal(e1.coord, e2.coord) & np.equal(e3.coord, e4.coord)).astype(float) * e1.coord
            self.point = None
            self.p_id = -1

    def add_face(self, e1, e2, e3, e4):
        face = PlasticBlock.FacePoint(e1, e2, e3, e4)
        face.point = self.add_vertex(face.coord[0], face.coord[1], face.coord[2])
        face.p_id = face.point.p_id
        self.faces.append(face)

        for e in [e1, e2, e3, e4]:
            e.add_face_e(face)

        return face

    class CtrPoint:
        # Control point id is defined by one-hot vector
        # 0 = + , 1 = - for x,y,z
        # ie: ctr_point 3 is 011 :  [1,-1,-1]
        def __init__(self, block, p_id, point, ex, ey, ez, curve, sub=2):
            self.p_id = p_id
            self.ex = ex
            self.ey = ey
            self.ez = ez
            self.coord = point.coord
            self.curve = curve
            # 3 free points array between control point and edge point
            self.x_list = []
            self.y_list = []
            self.z_list = []

            ## Todo  solve mixing of cylinder radius and sphere radius corners...

            # Radius consensus among 3 possibilities. (maximum influence is chosen, but could also be averaged)

            a_max = np.argmax([curve[0]*curve[1], ex.curve[0]*ex.curve[1], ey.curve[0]*ey.curve[1], ez.curve[0]*ez.curve[1]])
            self.r = [curve[0], ex.curve[0], ey.curve[0], ez.curve[0]][a_max]
            for i, r in enumerate([curve, ex.curve, ey.curve, ez.curve]):
                if i == a_max:
                    continue
                r[0] = 1
                r[1] = 0

            self.step_x = (1 / (sub - (1 if (self.r == 1) else 0))) * self.r
            self.step_y = (1 / (sub - (1 if (self.r == 1) else 0))) * self.r
            self.step_z = (1 / (sub - (1 if (self.r == 1) else 0))) * self.r
            if sub > 2:
                for i in range(1, sub-1):
                    self.x_list.append(block.add_vertex(self.coord[0]*(1 - self.step_x*i), self.coord[1], self.coord[2]).p_id)
                    self.y_list.append(block.add_vertex(self.coord[0], self.coord[1]*(1 - self.step_y*i), self.coord[2]).p_id)
                    self.z_list.append(block.add_vertex(self.coord[0], self.coord[1], self.coord[2]*(1 - self.step_z*i)).p_id)
            self.a_xy = self._build_quadrant(block, self.x_list, self.y_list, ex, ey, self.step_x, sub)
            self.a_xz = self._build_quadrant(block, self.x_list, self.z_list, ex, ez, self.step_y, sub)
            self.a_yz = self._build_quadrant(block, self.y_list, self.z_list, ey, ez, self.step_z, sub)

            for e in [ex, ey, ez]:
                # find value in control point coord at m_edge's 0 coord
                # (all m_edge have one 0 coord : ie [-1 0 1] is an m_edge
                if self.coord[np.where([e.coord == 0][0])[0][0]] < 0:
                    e.ctp_n = self
                else:
                    e.ctp_p = self

        # not to forget : each border of a quadrant array is overlapping the border of another quadrant array
        def _build_quadrant(self, block, e1_list, e2_list, e1, e2, step, sub=2):
            a = np.empty((sub, sub), dtype=int)
            if sub > 2:
                for i in range(1, sub-1):
                    a[0, i] = e2_list[i-1]
                    a[i, 0] = e1_list[i-1]
            facepoint = None
            for f1, f2 in product(e1.faces, e2.faces):
                if f1 == f2:
                    facepoint = f1
                    break
            a[ 0, 0] = self.p_id
            a[ 0,-1] = e2.p_id
            a[-1, 0] = e1.p_id
            a[-1,-1] = facepoint.p_id

            if sub > 2:
                for i in range(1, sub-1):
                    for j in range(1, sub - 1):
                        coord = self.coord - ((self.coord - e1.coord)*i*step + (self.coord - e2.coord)*j*step)
                        a[i, j] = block.add_vertex(coord[0], coord[1], coord[2]).p_id

                # get the inner edge (edge's mid point to face center)
                (a[-1, :])[1:-1] = e1.get_dual_edge(block, facepoint, step, sub)
                (a[:, -1])[1:-1] = e2.get_dual_edge(block, facepoint, step, sub)

            return a

    class Edge:
        def __init__(self, ctp_n, ctp_p):
            self.ctp_n = ctp_n
            self.ctp_p = ctp_p
            self.coord = (np.equal(ctp_n.coord, ctp_p.coord).astype(int) * ctp_p.coord).astype(float)
            self.p_id = -1
            self.point = None
            self.curve = []
            # self faces are the 2 that both points have in common
            self.faces = []
            self.dual_0 = []
            self.dual_1 = []

        def get_dual_edge(self, block, face, step, sub=2):
            if sub == 2:
                return []
            if face == self.faces[0]:
                sec = self.dual_0
            else:
                sec = self.dual_1
            coord = np.copy(self.coord)
            axis = np.where(self.coord - face.coord)[0][0]
            if len(sec) == 0:
                for i in range(sub - 2):
                    coord[axis] = self.coord[axis] * (1 - step*(i+1))
                    sec.append(block.add_vertex(coord[0], coord[1], coord[2]).p_id)
                if face == self.faces[0]:
                    self.dual_0 = sec
                else:
                    self.dual_1 = sec
            else:
                for i in range(sub - 2):
                    coord[axis] = self.coord[axis] * (1 - step*(i+1))
                    block.vertices[sec[i]] = ((block.vertices[sec[i]] + Vector(coord))/2)
            return sec

        def add_face_e(self, face):
            if face not in self.faces:
                self.faces.append(face)

    def get_edges(self):
        # use Quand-Edge structure to loop tru all unique edges and stitch them

        for e in self.m_edges.reshape(12):
            # consider only positive direction of m_edge
            ctp_n = e.ctp_n
            ctp_p = e.ctp_p
            for quadrant in [ctp_n.a_xy, ctp_n.a_xz, ctp_n.a_yz]:
                if quadrant[0, -1] == e.p_id:
                    edges_a = quadrant[:, -1][:]
                    self.edge_list.extend([[x, y] for x, y in zip(edges_a, edges_a[1:])])
                if quadrant[-1, 0] == e.p_id:
                    edges_a = quadrant[-1, :].T[:]
                    self.edge_list.extend([[x, y] for x, y in zip(edges_a, edges_a[1:])])

            for c in [ctp_n, ctp_p]:
                for quadrant in [c.a_xy, c.a_xz, c.a_yz]:
                    if quadrant[0, -1] == e.p_id:
                        edges_a = quadrant[0, :][:]
                        self.edge_list.extend([[x, y] for x, y in zip(edges_a, edges_a[1:])])
                        break
                    if quadrant[-1, 0] == e.p_id:
                        edges_a = quadrant[:, 0].T[:]
                        self.edge_list.extend([[x, y] for x, y in zip(edges_a, edges_a[1:])])
                        break

        # Convolution for each quadrant to list free edges
        if self.sub > 2:
            for c in self.control_points:
                for quadrant in [c.a_xy, c.a_xz, c.a_yz]:
                    self.quadrant_conv(quadrant)

    def quadrant_conv(self, quadrant):
        for i in range(1, self.sub - 1):
            for j in range(1, self.sub - 1):
                self.edge_list.append([quadrant[i][j - 1], quadrant[i][j]])
                self.edge_list.append([quadrant[i][j], quadrant[i - 1][j]])
                if j == self.sub - 2:
                    self.edge_list.append([quadrant[i][j], quadrant[i][j + 1]])
                if i == self.sub - 2:
                    self.edge_list.append([quadrant[i][j], quadrant[i + 1][j]])

    def get_faces(self):
        # Convolution to list all faces of each quadrant of each control point
        for c in self.control_points:
            for quadrant in [c.a_xy, c.a_xz, c.a_yz]:
                for i in range(1, self.sub):
                    for j in range(1, self.sub):
                        self.face_list.append([quadrant[i - 1][j - 1], quadrant[i - 1][j],
                                                quadrant[i][j], quadrant[i][j-1]])

    def deform_sphere(self, c: CtrPoint):
        c_id = c.p_id
        r = self.params[c_id, 0]
        mag = self.params[c_id, 1]
        if r == 0 or mag == 0:
            return
        control = self.vertices[c_id].copy()

        def _inrange(vertex, control):
            vc = self.vertices[vertex] - control
            if abs(vc[0]) <= r and abs(vc[1]) <= r and abs(vc[2]) <= r:
                return True
            return False
        # parameters should be a vector of length 40 reshaped into :
        # list of corner spherisations [r, magnitude]
        # for each corner, apply it's spherical transform
        vertices = []
        for quadrant in [c.a_xy, c.a_xz, c.a_yz]:
            vertices.extend(quadrant.reshape((1, -1)))

        center = self.vertices[c_id]*(1 - r)
        vertices = np.unique(vertices)
        for v in vertices:
            if _inrange(v, control):
                self.vertices[v] = self.round_corner(center, self.vertices[v], r, mag)

        for e in self.m_edges.reshape((1, -1))[0]:
            self.clean_dual_edge(e)

    def deform_cylinder(self, e: Edge):
        # Cylindracisations
        # for each edge, apply it's cylindrical transform [r, magnitude]
        e_id = e.p_id
        r = self.params[e_id, 0]
        mag = self.params[e_id, 1]
        if r == 0 or mag == 0:
            return
        mid = Vector(e.coord)
        mask = np.array(mid, dtype=bool)
        normal = Vector(np.invert(mask).astype(dtype=float))
        plane = np.where(mask)[0]

        def _inrange(vertex, ctrl):
            vc = self.vertices[vertex] - ctrl
            if abs(vc[0]) <= r and abs(vc[1]) <= r and abs(vc[2]) <= r:
                return True
            return False

        vertices = []
        for c in [e.ctp_n, e.ctp_p]:
            # filter quadrant normal to edge
            for quadrant in [[c.a_yz, c.a_xz, c.a_xy][i] for i in plane]:
                vertices.extend(quadrant.reshape((1, -1)))
        vertices = np.unique(vertices)

        c_p = (mid + normal)
        c_n = (mid - normal)
        for v in vertices:
            control = geometry.intersect_line_plane(c_p, c_n, self.vertices[v], normal)
            center = geometry.intersect_line_plane(normal, normal*-2, self.vertices[v], normal)
            center = center + (control - center)*(1 - r)
            if _inrange(v, control):
                self.vertices[v] = self.round_corner(center, self.vertices[v], r, mag)

        vertices = []
        for c in [e.ctp_n, e.ctp_p]:
            # filter quadrant of planar curve
            for quadrant in [[c.a_yz, c.a_xz, c.a_xy][i] for i in np.where(normal)[0]]:
                vertices.extend(quadrant[1:, 1:].reshape((1, -1)))
        vertices = np.unique(vertices)

        for v in vertices:
            control = geometry.intersect_line_plane(c_p, c_n, self.vertices[v], normal)
            center = geometry.intersect_line_plane(normal, normal*-2, self.vertices[v], normal)
            center = center + (control - center)*(1 - r)
            if (control - self.vertices[v]).length < (control - center).length:
                #continue
                self.vertices[v] = self.vertices[v] + (center - self.vertices[v])*.3*(center - self.vertices[v]).length

        for e in self.m_edges.reshape((1, -1))[0]:
            self.clean_dual_edge(e)

    def round_corner(self, origin: Vector, vector: Vector, r: float, magnitude: float):
        # project unto sphere of radius r at x0,y0,z0
        # mask ensures points on axis planes remain in that plane (numerical accuracy issue)
        mask = np.array(vector, dtype=bool)
        q = vector - origin
        if q.length == 0:
            return
        p = origin + q*(r/q.length)
        p = (p - vector)*magnitude + vector

        return Vector((p[0]*mask[0], p[1]*mask[1], p[2]*mask[2]))

    def clean_dual_edge(self, e):
        """
            Get a dual list from an edge
            find the 2 quadrants sharing it
        """
        if self.sub > 2:
            for dual in [e.dual_0, e.dual_1]:
                quads = []
                for c in [e.ctp_n, e.ctp_p]:
                    for quadrant in [c.a_xy, c.a_xz, c.a_yz]:
                        if dual[0] == quadrant[-1,1]:
                            quads.append(quadrant)
                            break
                        if dual[0] == quadrant[1,-1]:
                            quads.append(quadrant.T)
                            break
                for i in range(self.sub - 1):
                    a = Vector(e.coord)#self.vertices[quads[0][-1][i]]
                    b = self.vertices[quads[0][-2][i]]
                    c = self.vertices[quads[1][-2][i]]
                    mask = np.array(a, dtype=bool)
                    n = np.invert(mask).astype(dtype=float)
                    p = geometry.intersect_line_plane(c, b, a, Vector((n[0], n[1], n[2])))
                    self.vertices[quads[0][-1][i]] = Vector((p[0]*mask[0], p[1]*mask[1], p[2]*mask[2]))

    def apply_affine(self, affine_params):
        # get 12 parameters of an affine transformation matrix
        # [scale x, y, z, shear xy, xz, yx, yz, zx, zy, rotation x, y, z]
        S = Matrix.Identity(3)

        # scale x, y, z
        S[0][0] = max(0.05, affine_params[0])
        S[1][1] = max(0.05, affine_params[1])
        S[2][2] = max(0.05, affine_params[2])

        # shear
        S[1][0] = affine_params[3]
        S[2][0] = affine_params[4]
        S[0][1] = affine_params[5]
        S[1][2] = affine_params[6]
        S[0][2] = affine_params[7]
        S[1][2] = affine_params[8]

        Rx = Matrix.Rotation(radians(affine_params[9]), 3, 'X')
        Ry = Matrix.Rotation(radians(affine_params[10]), 3, 'Y')
        Rz = Matrix.Rotation(radians(affine_params[11]), 3, 'Z')
        R = Rx @ Ry @ Rz

        T = R @ S
        for i in range(len(self.vertices)):
            self.vertices[i] = T @ self.vertices[i]

    # Returns a triangulated obj array format
    def get_obj(self):
        # get edge list
        # get face list
        # get vertex list
        vertices = np.zeros((len(self.vertices), 3))
        for i in range(len(self.vertices)):
            vertices[i, :] = [self.vertices[i].x, self.vertices[i].y, self.vertices[i].z]

        edges = self.edge_list.copy()
        faces = []
        # triangulate the quadmesh
        for f in range(len(self.face_list)):
            faces.append([self.face_list[f][0], self.face_list[f][1], self.face_list[f][2]])
            faces.append([self.face_list[f][2], self.face_list[f][3], self.face_list[f][0]])
            edges.append([self.face_list[f][0], self.face_list[f][2]])

        return vertices, np.asarray(edges), np.asarray(faces)

    def random_sample(self, eccentricity: int, rot_ecc: int, rot_mag: float, shear_ecc: int, shear_mag):
        # Eccentricity is parameter of block's variance away from a primitive
        # defined as eccentricity : #of defining components
        # return the parameter vector
        shear_ecc = min(shear_ecc, 9)
        rot_ecc = min(rot_ecc, 3)
        rot_mag = min(rot_mag, 1) * 89.9

        shear = np.random.uniform(0.05, shear_mag, shear_ecc)
        shear = np.append(shear, np.zeros(9 - shear_ecc))
        np.random.shuffle(shear)
        (shear[:3])[shear[:3] == 0] = 1

        rotation = np.random.uniform(0., rot_mag, rot_ecc)
        rotation = np.append(rotation, np.zeros(3 - rot_ecc))
        np.random.shuffle(rotation)

        affine = np.append(shear, rotation)

        params = np.zeros((20, 2))
        params[:, 0] = 1
        if eccentricity == 0:
            params[:8, 0] = 1

            return np.append(params.reshape((-1, )), affine)
        if eccentricity >= 4:

            return np.append(np.random.uniform(0, 1, 40), affine)
        cube = params
        sphere = params.copy()
        cylinder = params.copy()
        if eccentricity == 1:
            cube[:8, 0] = 1
            sphere[:8, :] = 1
            ne = random.randint(0, 2)
            cylinder[8+(ne*4):12+(ne*4), :] = 1

        if eccentricity == 2:
            cube[:8, 0] = 1
            sphere[:8, 0] = random.uniform(0.05, 1)
            sphere[random.randint(0, 7), :] = 1
            cylinder[8:20, 0] = random.uniform(0.05, 1)
            cylinder[random.randint(0, 12), :] = 1

        if eccentricity == 3:
            r = random.uniform(0.5, 1)
            cube[:8, 0] = r/4
            cube[:8, 1] = 1
            nc = random.randint(0, 8)
            sphere[nc, 0] = r
            sphere[nc, 1] = 1
            cylinder[8:, 0] = np.random.uniform(0.1, 1, 12)
            cylinder[8:, 1] = 1

        params = [cube, sphere, cylinder][random.randint(0, 2)].reshape((40, ))
        return np.append(params, affine)




def main():
    foo = PlasticBlock()


if __name__ == "__main__":
    main()
