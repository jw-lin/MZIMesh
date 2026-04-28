### What is this?
`MZIMesh` is a simple package for simulating meshes of Mach-Zehnder interferometers in monochromatic light. Currently, you can set

1. biases for each $\phi$ and $\theta$ phase shifter
2. uneven directional coupler split ratios
3. a thermal crosstalk matrix

See the AstroPIC folder from some examples.

### Install
```
pip install git+https://github.com/jw-lin/MZIMesh.git
```

### Quickstart

Make a $5\times 5$ mesh:

```
from MZIMesh import TriMesh
mesh = TriMesh(5)
```

Interactive plot:

```
mesh.live_plot()
```

Set individual mesh parameters. The `mesh` works as a list of individual MZIs in ascending diagonal order.

```
mesh[0].phi0 = 1. # rad
mesh[3].theta0 = -1. # rad
```

Actuate a phase shifter

```
mesh[2].Vphi = 3.
mesh[2].Vtheta = 2.
```

Propagate

```
launch_field = mesh.make_launch_field("flat") # flat-topped piston
intensities = mesh.measure(launch_field)
```
