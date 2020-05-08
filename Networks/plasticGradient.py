"""
Compute gradient manually stepping all parameters
    :parameter
        V,F vertices nx3 and index of triangular faces of a mesh
        b_params: parameters of PlasticBlock
        epsilon : size of step
"""
import numpy as np
import igl
from PlasticBlock import PlasticBlock


def plasticGradient(V, F, b_params, epsilon=0.001):

    block = PlasticBlock(b_params)
    Vb, _, Fb = block.get_obj()
    loss = igl.hausdorff(V, F, Vb, Fb)
    m_gradient = np.zeros((b_params.shape[0],))

    for p in range(len(b_params)):
        # generate alternate mesh
        param = b_params.copy()
        param[p] = param[p] + epsilon
        block = PlasticBlock(param)
        # compute distance
        Vb, _, Fb = block.get_obj()
        d = igl.hausdorff(V, F, Vb, Fb)
        m_gradient[p] = (d - loss)/epsilon

    return m_gradient
