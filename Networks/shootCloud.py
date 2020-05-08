import point_cloud_utils as pcu
import numpy as np
from scipy.spatial import KDTree

from PlasticBlock import PlasticBlock


def shootCloud(V, F, density: int = 256, occlusion = 80, shuffle=False):
    cloud, _ = pcu.sample_mesh_random(V, F, np.array([], dtype=V.dtype), num_samples=density+occlusion)

    tree = KDTree(cloud)

    #find occlusion nearest neighbours and remove
    if occlusion > 0:
        x = np.random.randint(len(cloud))
        _, nearest_ind = tree.query(cloud[x].reshape(-1, 3), k=occlusion)
        cloud = np.delete(cloud, nearest_ind, axis=0)

    #unorder the data
    if shuffle :
        np.random.shuffle(cloud)

    return cloud

def write_objs():
    name = "plastic_bloc_"

    for i in range(10):
        block = PlasticBlock(eccentricity=4,rot_ecc=0,rot_mag=0,shear_ecc=0,shear_mag=0)
        V, E, F = block.get_obj()
        cloud = shootCloud(V, F)
        pcu.write_obj(name+str(i)+".obj",V,F,V)
        pcu.write_obj(name + str(i)+"-cloud.obj", cloud, F, V)


def main():
    foo = PlasticBlock()
    write_objs()

if __name__ == "__main__":
    main()
