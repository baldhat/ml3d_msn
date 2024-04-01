## MSN: Ablation Study On Morphing and Sampling Network for Dense Point Cloud Completion Paper

![image](https://github.com/baldhat/ml3d_msn/assets/42282389/f9ea9852-36ad-4b9c-a645-27fd6fb8bf40)

[[paper]](http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf) [[data]](https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7?usp=sharing)

MSN is a learning-based shape completion method which can preserve the known structures and generate dense and evenly distributed point clouds. See our AAAI 2020 [paper](http://cseweb.ucsd.edu/~mil070/projects/AAAI2020/paper.pdf) for more details.

In this project, we also provide an implementation for the Earth Mover's Distance (EMD) of point clouds, which is based on the auction algorithm and only needs $O(n)$ memory.

![](/teaser.png)
*with 32,768 points after completion*


### Usage

#### 1) Envrionment & prerequisites

- Pytorch 1.2.0
- CUDA 10.0
- Python 3.7
- [Visdom](https://github.com/facebookresearch/visdom)
- [Open3D](http://www.open3d.org/docs/release/index.html#python-api-index)

#### 2) Compile

Compile our extension modules:  

    cd emd
    python3 setup.py install
    cd expansion_penalty
    python3 setup.py install
    cd MDS
    python3 setup.py install
    cd metrics/CD/chamfer3D
    python3 setup.py install

#### 3) Download data and trained models

Download the data and trained models from [here](https://drive.google.com/drive/folders/1X143kUwtRtoPFxNRvUk9LuPlsf1lLKI7?usp=sharing).  We don't provide the partial point clouds of the training set due to the large size. If you want to train the model, you can generate them with the [code](https://github.com/wentaoyuan/pcn/tree/master/render) and [ShapeNetCore.v1](https://shapenet.org/). We generate 50 partial point clouds for each CAD model.

#### 4) Train or validate

Run `python3 val.py` to validate the model or `python3 train.py` to train the model from scratch.

#### 5) Ablation 
Data Processing with Perlin Noise, Gaussian Noise and Outlier. For Outliers, we added 20 outliers to each point cloud. The outliers are uniformly sampled within the unit cube.

Encoder options include PointNet++, PointConv, PointTransformer, and FusedEncoder. Our architecture employs a fused design, integrating additional multi-layer perceptrons to merge features from PointConv and PointNet++.

Original MSN paper adapts EMD as their loss function. However, using EMD as a loss function could be expensive since it enforces a one-to-one mapping between two point sets. Our architecture uses Density Aware Chamfer Distance (DCD) as a loss function for faster training while maintaining a good result. Furthermore, DCD provides a variant to deal with point sets with different numbers of points. Hence, we could potentially train a more dense point cloud completion. DCD takes a step from CD and attempts to provide a rationale bridge towards EMD for a better sense of point distribution rather than being blinded by its nearest neighbour.

Include a **CUDA** version, and a **PYTHON** version with pytorch standard operations.
NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt thresholds accordingly.

- [x] F - Score  
### Citation

If you find our work useful for your research, please cite:
```
@article{liu2019morphing,
  title={Morphing and Sampling Network for Dense Point Cloud Completion},
  author={Liu, Minghua and Sheng, Lu and Yang, Sheng and Shao, Jing and Hu, Shi-Min},
  journal={arXiv preprint arXiv:1912.00280},
  year={2019}
}
```

### License

This project Code is released under the Apache License 2.0 (refer to the LICENSE file for details).
