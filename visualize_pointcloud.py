import open3d as o3d
import numpy as np


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


# pc = read_pcd("./data/complete/04256520_a92f6b7dccd0421f7248d9dbed7a7b8.pcd")
pc = read_pcd("./data/modelnet10/point_clouds/complete/bed_train_bed_0468.pcd")
points = np.asarray(pc.points)
print(points.shape)
print(np.max(points))
print(np.min(points))
o3d.visualization.draw_geometries([pc])