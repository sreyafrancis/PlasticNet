# PlasticNet - IFT 6132 Project
Structured prediction of primitive shapes in point clouds

This is based on papers:
*Structured Prediction Energy Network* <https://arxiv.org/abs/1511.06350>

*End-to-End Learning for Structured Prediction Energy Networks* <https://arxiv.org/abs/1703.05667>

*PointNet++* <https://arxiv.org/abs/1706.02413>

*Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs* <http://arxiv.org/abs/1711.09869>

*Point Cloud Oversegmentation with Graph-Structured Deep Metric Learning* <https://arxiv.org/pdf/1904.02113>.

### Plastic Net Architecture
<img src="https://github.com/sreyafrancis/PlasticNet/blob/master/images/SPEN_Plastic.JPG" width="500">

### Point cloud segmentation 
<img src="http://imagine.enpc.fr/~simonovm/largescale/teaser.jpg" width="500">

<img src="http://recherche.ign.fr/llandrieu/SPG/ssp.png" width="500">

## Requirements

*1.* Install [PyTorch](https://pytorch.org) and [torchnet](https://github.com/pytorch/tnt).
```
pip install git+https://github.com/pytorch/tnt.git@master
``` 

*2.* Install additional Python packages:
```
pip install future python-igraph tqdm transforms3d pynvrtc fastrlock cupy h5py sklearn plyfile scipy
```

*3.* Install Boost (1.63.0 or newer) and Eigen3, in Conda:<br>
```
conda install -c anaconda boost; conda install -c omnia eigen3; conda install eigen; conda install -c r libiconv
```

*4.* Make sure that cut pursuit was downloaded. Otherwise, clone [this repository](https://github.com/loicland/cut-pursuit) or add it as a submodule in `/partition`: <br>
```
cd partition
git submodule init
git submodule update --remote cut-pursuit
```

*5.* Compile the ```libply_c``` and ```libcp``` libraries:
```
CONDAENV=YOUR_CONDA_ENVIRONMENT_LOCATION
cd partition/ply_c
cmake . -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
cd ..
cd cut-pursuit
mkdir build
cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
*6.* (optional) Install [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)

*7.* Install [libgl library](https://libigl.github.io/) 

```output_type``` defined as such:
- ```'i'``` = input rgb point cloud
- ```'g'``` = ground truth (if available), with the predefined class to color mapping
- ```'f'``` = geometric feature with color code: red = linearity, green = planarity, blue = verticality
- ```'p'``` = partition, with a random color for each superpoint
- ```'r'``` = result cloud, with the predefined class to color mapping
- ```'e'``` = error cloud, with green/red hue for correct/faulty prediction 
- ```'s'``` = superedge structure of the superpoint (toggle wireframe on meshlab to view it)




