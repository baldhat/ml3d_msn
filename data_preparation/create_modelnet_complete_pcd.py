import trimesh
import open3d as o3d
import os
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

data_path = "../data/modelnet10/ModelNet10"
output_path = "../data/modelnet10/point_clouds/complete"
if not os.path.exists(output_path):
    os.mkdir(output_path)

for model_category in os.listdir(data_path):
    for split in ["train", "test"]:
        for model_id in os.listdir(os.path.join(data_path, model_category, split)):
            if ".off" in model_id:
                mesh = trimesh.load_mesh(os.path.join(data_path, model_category, split, model_id))
                mesh.apply_scale(1.0 / max(mesh.extents))
                pcd = PointCloud()
                points, _ = trimesh.sample.sample_surface(mesh, 10000)
                pcd.points = Vector3dVector(points)
                o3d.io.write_point_cloud(os.path.join(output_path,
                                                      model_category + "_" + split + "_" + model_id.split(".")[0] + ".pcd"), pcd)