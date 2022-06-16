# Used for visualization purpose, not much of use elsewhere so not maintained.
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

class Visualization:
    def leftbuildingvisualize(self,leftbuild,cloud):
        cloud[0].paint_uniform_color([0, 0, 0])
        cloud[1].paint_uniform_color([0, 0, 1])
        cloud[2].paint_uniform_color([1, 0, 0])
        cloud[3].paint_uniform_color([0, 1, 1])
        cloud[4].paint_uniform_color([0, 0.75, 0.25])
        cloud[5].paint_uniform_color([0, 0.25, 0.75])
        cloud[6].paint_uniform_color([1, 0, 0])
        cloud[7].paint_uniform_color([0, 0.5, 1])
        o3d.visualization.draw_geometries([leftbuild, cloud[0], cloud[1], cloud[2], cloud[3], cloud[4], cloud[5], cloud[6], cloud[7]],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])
    def rightbuildingvisualize(self,rightbuild,cloud):
        cloud[0].paint_uniform_color([0, 0, 0])
        cloud[1].paint_uniform_color([0, 0, 1])
        cloud[2].paint_uniform_color([1, 0, 0])
        cloud[3].paint_uniform_color([0, 1, 1])
        cloud[4].paint_uniform_color([0, 0.75, 0.25])
        cloud[5].paint_uniform_color([0, 0.25, 0.75])
        cloud[6].paint_uniform_color([1, 0, 0])
        cloud[7].paint_uniform_color([0, 0.5, 1])
        o3d.visualization.draw_geometries([rightbuild, cloud[0], cloud[1], cloud[2], cloud[3], cloud[4], cloud[5], cloud[6], cloud[7]],
                                        zoom=0.8,
                                        front=[-0.4999, -0.1659, -0.8499],
                                        lookat=[2.1813, 2.0619, 2.0999],
                                        up=[0.1204, -0.9852, 0.1215])