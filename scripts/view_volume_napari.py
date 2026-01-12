import numpy as np
import napari

AI = np.load(r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data\processed\inversion_volume\AI_pred_cube_masked.npy")
FAC = np.load(r"H:\GK-MRL-PhysicsConsistent-Inversion\GK-MRL-PhysicsConsistent-Inversion\data\processed\inversion_volume\Facies_pred_cube.npy")

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(AI, name="AI_pred", rendering="mip")           # 可切片 + 3D 视图
viewer.add_labels(FAC.astype(np.int32), name="Facies_pred")     # 0..3, -1 会显示为背景
napari.run()