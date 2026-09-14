import meep as mp
import matplotlib.pyplot as plt

print("MEEP version:", mp.__version__)

cell = mp.Vector3(2, 2, 0)

geometry = [
    mp.Cylinder(
        radius=0.25,
        material=mp.Medium(epsilon=4)
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    resolution=50
)

sim.plot2D()

plt.savefig("meep_test.png", dpi=200)
plt.show()