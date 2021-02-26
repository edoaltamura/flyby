import os
import unyt
import h5py
import numpy as np
import swiftsimio as sw
from swiftsimio.visualisation.projection import project_gas
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from PIL import Image, ImageOps, ImageEnhance

import numba
numba.config.NUMBA_NUM_THREADS = 28

def binary_normalise(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def force_aspect(axes, aspect=1):
    im = axes.get_images()
    extent = im[0].get_extent()
    axes.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def process_single_halo(
        path_to_snap: str,
        path_to_catalogue: str,
        slice_thickness: unyt.unyt_quantity = unyt.unyt_quantity(0.1, unyt.Mpc),
        map_size_R500_units: int = 3,
        field: str = 'densities'
):
    # Retrieve VR properties using zoom information
    with h5py.File(path_to_catalogue, 'r') as vr_file:
        M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
        R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
        R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][0], unyt.Mpc)
        xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
        yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
        zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)

    map_size = map_size_R500_units * R500c

    # Construct spatial mask to feed into swiftsimio
    mask = sw.mask(path_to_snap, spatial_only=False)
    region = [
        [xCen - 6 * R500c, xCen + 6 * R500c],
        [yCen - 6 * R500c, yCen + 6 * R500c],
        [zCen - slice_thickness, zCen + slice_thickness]
    ]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", 1.e5 * mask.units.temperature, 5.e9 * mask.units.temperature)
    data = sw.load(path_to_snap, mask=mask)
    region = [
        xCen - map_size,
        xCen + map_size,
        yCen - map_size,
        yCen + map_size
    ]

    if field == 'densities':
        smoothed_map = project_gas(data, resolution=resolution, project="densities", parallel=True, region=region)
        unitLength = data.metadata.units.length
        unitMass = data.metadata.units.mass
        rho_crit = unyt.unyt_quantity(
            data.metadata.cosmology_raw['Critical density [internal units]'],
            unitMass / unitLength ** 3
        ).to('Msun/Mpc**3')
        smoothed_map /= rho_crit.value

    elif field == 'mass_weighted_temperatures':
        mass_map = project_gas(data, resolution=resolution, project="masses", parallel=True, region=region)
        data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures
        mass_weighted_temp_map = project_gas(data, resolution=resolution, project="mass_weighted_temperatures",
                                             parallel=True, region=region)
        smoothed_map = mass_weighted_temp_map / mass_map

    smoothed_map[smoothed_map == 0.] = np.nan
    smoothed_map = binary_normalise(np.log10(smoothed_map))

    # Set-up figure and axes instance
    fig = plt.figure(figsize=(8, 8), dpi=resolution // 8)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.imshow(smoothed_map, origin="lower", extent=region)
    plt.show()
    fig.savefig(f'{field}.png', bbox_inches='tight', pad_inches=0.)


if __name__ == "__main__":
    resolution = 4096
    snap_filepath_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/snapshots/L0300N0564_VR813_+1res_MinimumDistance_2749.hdf5"
    velociraptor_properties_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/stf/L0300N0564_VR813_+1res_MinimumDistance_2749/L0300N0564_VR813_+1res_MinimumDistance_2749.properties"

    process_single_halo(
        snap_filepath_zoom,
        velociraptor_properties_zoom,
        slice_thickness=unyt.unyt_quantity(0.1, unyt.Mpc),
        map_size_R500_units=3,
        field='densities'
    )