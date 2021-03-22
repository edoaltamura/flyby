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
    mask = sw.mask(path_to_snap, spatial_only=True)
    region = [
        [xCen - 1.5 * map_size, xCen + 1.5 * map_size],
        [yCen - 1.5 * map_size, yCen + 1.5 * map_size],
        [zCen - slice_thickness, zCen + slice_thickness]
    ]
    # temperature_units = mask.units.temperature
    # density_units = mask.units.mass / mask.units.length ** 3
    # density_low = (1e-10 / unyt.cm ** 3 * unyt.mp).to(density_units)
    # density_high = (2 / unyt.cm ** 3 * unyt.mp).to(density_units)
    mask.constrain_spatial(region)
    # mask.constrain_mask("gas", "temperatures", 1.e5 * temperature_units, 5.e9 * temperature_units)
    # mask.constrain_mask("gas", "densities", density_low, density_high)
    data = sw.load(path_to_snap, mask=mask)
    region = [
        xCen - map_size,
        xCen + map_size,
        yCen - map_size,
        yCen + map_size
    ]

    if field == 'densities':
        smoothed_map = project_gas(data, resolution=resolution, project="densities", parallel=True, region=region)
        cmap = 'bone'

    elif field == 'mass_weighted_temperatures':
        mass_map = project_gas(data, resolution=resolution, project="masses", parallel=True, region=region)
        data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures
        mass_weighted_temp_map = project_gas(data, resolution=resolution, project="mass_weighted_temperatures",
                                             parallel=True, region=region)
        smoothed_map = mass_weighted_temp_map / mass_map
        cmap = 'gist_heat'

    elif field == 'velocity_divergences':
        data.gas.velocity_divergences[data.gas.velocity_divergences.value >= 0] = 0
        data.gas.velocity_divergences = np.abs(data.gas.velocity_divergences)
        smoothed_map = project_gas(data, resolution=resolution, project="velocity_divergences", parallel=True, region=region)
        cmap = 'pink'

    elif field == 'entropies':
        mass_map = project_gas(data, resolution=resolution, project="masses", parallel=True, region=region)
        data.gas.mass_weighted_temperatures = data.gas.masses * data.gas.temperatures

        mean_molecular_weight = 0.59
        data.gas.entropies = (
                data.gas.mass_weighted_temperatures *
                unyt.boltzmann_constant * unyt.pm * mean_molecular_weight /
                data.gas.densities.to('g/cm**3') ** (2 / 3)
        )
        entropy_map = project_gas(data, resolution=resolution, project="entropies", parallel=True, region=region)
        smoothed_map = entropy_map / mass_map
        cmap = 'copper'

    # smoothed_map[smoothed_map == 0.] = np.nan
    # smoothed_map = binary_normalise(np.log10(smoothed_map))

    # Set-up figure and axes instance
    fig = plt.figure(figsize=(8, 8), dpi=resolution // 8)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(0, 0, 1, 1)
    # plt.axis('off')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    force_aspect(ax)
    ax.imshow(smoothed_map, origin="lower", extent=region, cmap=cmap)
    # ax.set_xlim(region[0], region[1])
    # ax.set_ylim(region[2], region[3])
    # circle_r500 = plt.Circle((xCen, yCen), R500c, color="black", fill=False, linestyle='-')
    # ax.add_artist(circle_r500)
    # ax.text(
    #     xCen,
    #     yCen + 1.05 * R500c,
    #     r"$R_{500c}$",
    #     color="black",
    #     ha="center",
    #     va="bottom"
    # )
    plt.show()
    fig.savefig(f'{field}.png', bbox_inches='tight', pad_inches=0.)


if __name__ == "__main__":
    resolution = 256
    snap_filepath_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth/snapshots/L0300N0564_VR2414_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036.hdf5"
    velociraptor_properties_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR2414_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth/stf/L0300N0564_VR2414_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036/L0300N0564_VR2414_+1res_MinimumDistance_fixedAGNdT8.5_Nheat1_SNnobirth_0036.properties"

    for field in ['densities']:#, 'mass_weighted_temperatures', 'velocity_divergences', 'entropies']:
        print(field)
        process_single_halo(
            snap_filepath_zoom,
            velociraptor_properties_zoom,
            slice_thickness=unyt.unyt_quantity(1, unyt.Mpc),
            map_size_R500_units=3,
            field=field
        )
