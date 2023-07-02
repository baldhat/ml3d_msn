import open3d as o3d
import numpy as np


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


# pc = read_pcd("./data/complete/04256520_a92f6b7dccd0421f7248d9dbed7a7b8.pcd")
pc1 = read_pcd("./data/modelnet10/point_clouds/train/pcd/bathtub_0105/1.pcd")
pc2 = read_pcd("./data/modelnet10/point_clouds_noisy/train/pcd/bathtub_0105/1.pcd")
points1 = np.asarray(pc1.points)
points2 = np.asarray(pc2.points) + np.array([1.2, 0.0, 0.0])
pc2.points = o3d.utility.Vector3dVector(points2)
o3d.visualization.draw_geometries([pc1, pc2])
