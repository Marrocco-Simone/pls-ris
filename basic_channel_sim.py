import sionna.rt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
no_preview = True

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, mi


scene = load_scene("mesh_scene/scene.xml")

scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=2,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = PlanarArray(num_rows=2,
                             num_cols=2,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

tx = Transmitter("tx", position=mi.Point3f([0, -8, 2]), orientation=mi.Point3f([0, 0, 0]))
rx = Receiver("rx", position=mi.Point3f([0, 8, 2]), orientation=mi.Point3f([0, 180, 0]))

scene.add(tx)
scene.add(rx)


my_cam = Camera(position=mi.Point3f([-250,250,150]), look_at=[-15,30,28])
scene.render(camera=my_cam, resolution=(650, 500), num_samples=512)


# Ray tracing
scene.frequency = 3.5e9

p_solver  = PathSolver()
paths = p_solver(scene=scene,
                 max_depth=5,
                 los=True,
                 specular_reflection=True,
                 diffuse_reflection=False,
                 refraction=True,
                 synthetic_array=False, #true per versione semplificata di default ma va bene così
                 seed=41)
if no_preview:
    scene.render(camera=my_cam, paths=paths, clip_at=20)
else:
    scene.preview(paths=paths, clip_at=20)

a, tau = paths.cir(normalize_delays=True, out_type="numpy")

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
print("Shape of a: ", a.shape) # type: ignore

# Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
print("Shape of tau: ", tau.shape)

# Per ottenere la matrice di canale H sommiamo i guadagni dei percorsi sulla penultima dimensione (num_paths)
h = tf.reduce_sum(a, axis=-2)

# Riduco le dimensioni per una singola coppia Tx/Rx
h = tf.squeeze(h)

print(f"\nMatrice di Canale H ottenuta (forma: [num_rx_ant, num_tx_ant]):")
print(h.numpy())
print(f"Forma di H: {h.shape}")