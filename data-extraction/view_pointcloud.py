import open3d as o3d
import numpy as np


def open_pointcloud(ply_path):
    print("Opening:", ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)

    if pcd.is_empty():
        print("Pointcloud kosong atau gagal dibaca.")
        return

    points = np.asarray(pcd.points)

    print("Number of points:", len(points))
    print("X min/max:", points[:, 0].min(), points[:, 0].max())
    print("Y min/max:", points[:, 1].min(), points[:, 1].max())
    print("Z min/max:", points[:, 2].min(), points[:, 2].max())

    # Optional: downsample supaya lebih ringan
    pcd_down = pcd.voxel_down_sample(voxel_size=0.005)
    # pcd_down = pcd

    print("After downsample:", len(pcd_down.points))

    # Tambahkan koordinat axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=[0, 0, 0]
    )

    o3d.visualization.draw_geometries(
        [pcd_down, axis],
        window_name="Pointcloud Viewer",
        width=1280,
        height=720,
        point_show_normal=False
    )


if __name__ == "__main__":
    open_pointcloud("./data-extraction/pointcloud_wls.ply")