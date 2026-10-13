import numpy as np
import pymeshlab
import polyscope as ps

ms = pymeshlab.MeshSet()

ms.load_new_mesh('final_star.obj')
m = ms.current_mesh()
star_v, star_f = m.vertex_matrix(), m.face_matrix()

ms.load_new_mesh('final_scan.obj')
m = ms.current_mesh()
scan_v, scan_f = m.vertex_matrix(), m.face_matrix()

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("star", star_v, star_f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )
ps.register_surface_mesh("scan", scan_v, scan_f,
                         color=np.array([255, 164, 0]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

# ps.set_view_projection_mode("orthographic")  # orthographic  perspective
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
ps.set_shadow_blur_iters(3)
ps.set_shadow_darkness(0.5)
ps.set_up_dir("y_up")
# ps.set_front_dir('x_front')
ps.set_SSAA_factor(4)
ps.show()
