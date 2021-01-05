import matplotlib

matplotlib.use('Agg')

import unyt
import h5py
import numpy as np
from typing import Union
import swiftsimio as sw
from swiftsimio.visualisation.sphviewer import SPHViewerWrapper
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image, ImageEnhance
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

resolution = 4096
snap_filepath_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/snapshots/L0300N0564_VR813_+1res_MinimumDistance_2749.hdf5"
velociraptor_properties_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/stf/L0300N0564_VR813_+1res_MinimumDistance_2749/L0300N0564_VR813_+1res_MinimumDistance_2749.properties"

color_bounds_log10density = [0., 8.]
color_bounds_log10temperature = [5., 9.]


def binary_normalise(array: np.ndarray, bin_min: Union[float, None] = None, bin_max: Union[float, None] = None):
    if bin_min is None:
        bin_min = np.min(array)
    if bin_max is None:
        bin_max = np.max(array)

    return (array - bin_min) / (bin_max - bin_min)


def gamma_correction(img: np.ndarray, gamma: float = 1.0):
    igamma = 1.0 / gamma
    imin, imax = img.min(), img.max()

    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** igamma
    img_c = img_c * (imax - imin) + imin
    return img_c


def arcsinh_stretching(image_lrgb: np.ndarray, beta: float = 0.1):
    input_img = image_lrgb.copy()
    np.true_divide(input_img, beta, out=input_img)
    np.arcsinh(input_img, out=input_img)
    np.true_divide(input_img, np.arcsinh(1. / beta), out=input_img)
    return input_img


def combine_layers(
        dataset1: np.ndarray,
        dataset2: np.ndarray,
        res: int,
        cmap1: str = 'Greys',
        cmap2: str = 'coolwarm',
) -> np.ndarray:
    cmap1 = cm.get_cmap(cmap1)
    imgobj = Image.fromarray(np.uint8(cmap1(dataset1) * 255)).convert('RGB')
    data = imgobj.getdata()  # Returns an object that can be turned into list()
    bin_dens_map = np.array(list(data), dtype=np.int) / 255

    cmap2 = cm.get_cmap(cmap2)
    imgobj = Image.fromarray(np.uint8(cmap2(dataset2) * 255)).convert('RGB')
    data = imgobj.getdata()
    bin_temp_map = np.array(list(data), dtype=np.int) / 255

    # Initialise combined RGBA image
    combined_map = np.zeros((res * res, 4), dtype=np.float)

    # Perform the mixing and set weights
    combined_map[:, 0] += bin_temp_map[:, 0] * gamma_correction((1 - bin_dens_map[:, 0]), gamma=1.2)
    combined_map[:, 1] += bin_temp_map[:, 1] * gamma_correction((1 - bin_dens_map[:, 0]), gamma=1.3)
    combined_map[:, 2] += bin_temp_map[:, 2] * gamma_correction((1 - bin_dens_map[:, 0]), gamma=1.4)
    combined_map[:, 3] += gamma_correction(bin_dens_map[:, 0], gamma=1.4)

    # Apply channel stretching
    # combined_map[:, 3] = binary_normalise(combined_map[:, 3])
    combined_map = arcsinh_stretching(combined_map)

    # Do further enhancement with the Pillow library
    combined_map = combined_map.reshape((res, res, 4))
    imgobj = Image.fromarray(np.uint8(combined_map * 255), 'RGBA')
    imgobj = ImageEnhance.Sharpness(imgobj).enhance(1.3)
    imgobj = ImageEnhance.Contrast(imgobj).enhance(1.4)
    imgobj = ImageEnhance.Color(imgobj).enhance(1.3)
    data = imgobj.getdata()
    combined_map = np.array(list(data), dtype=np.int) / 255
    combined_map = combined_map.reshape((res, res, 4))

    return combined_map


def make_image(part_type_data, project: str = "masses", camera_kwargs=None):
    sph_map = SPHViewerWrapper(part_type_data, smooth_over=project)
    sph_map.quick_view(**camera_kwargs)
    return sph_map.image


def force_aspect(axes, aspect=1):
    im = axes.get_images()
    extent = im[0].get_extent()
    axes.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


# Retrieve VR properties using zoom information
with h5py.File(velociraptor_properties_zoom, 'r') as vr_file:
    M200c = unyt.unyt_quantity(vr_file['/Mass_200crit'][0] * 1e10, unyt.Solar_Mass)
    R200c = unyt.unyt_quantity(vr_file['/R_200crit'][0], unyt.Mpc)
    R500c = unyt.unyt_quantity(vr_file['/SO_R_500_rhocrit'][0], unyt.Mpc)
    xCen = unyt.unyt_quantity(vr_file['/Xcminpot'][0], unyt.Mpc)
    yCen = unyt.unyt_quantity(vr_file['/Ycminpot'][0], unyt.Mpc)
    zCen = unyt.unyt_quantity(vr_file['/Zcminpot'][0], unyt.Mpc)

size = unyt.unyt_quantity(6., unyt.Mpc)
print(xCen, yCen, zCen, size)

# Construct spatial mask to feed into swiftsimio
mask = sw.mask(snap_filepath_zoom, spatial_only=False)
region = [
    [xCen - size, xCen + size],
    [yCen - size, yCen + size],
    [zCen - size, zCen + size]
]
mask.constrain_spatial(region)
mask.constrain_mask(
    "gas",
    "temperatures",
    10 ** color_bounds_log10temperature[0] * mask.units.temperature,
    10 ** color_bounds_log10temperature[1] * mask.units.temperature
)
data = sw.load(snap_filepath_zoom, mask=mask)
region = [
    xCen - 0.8 * size,
    xCen + 0.8 * size,
    yCen - 0.8 * size,
    yCen + 0.8 * size
]

# Calculate particle mass and rho_crit
unitLength = data.metadata.units.length
unitMass = data.metadata.units.mass
rho_crit = unyt.unyt_quantity(
    data.metadata.cosmology_raw['Critical density [internal units]'],
    unitMass / unitLength ** 3
).to('Msun/Mpc**3')

#  Multiply the values of the smoothing length by the kernel gamma (~2.0)
data.gas.smoothing_lengths = data.gas.smoothing_lengths * 2.0

# Define camera path and number of frames
num_frames = 1
rotation_path = np.linspace(0, 360, num_frames)
radial_path = np.sqrt(np.linspace(0.2 ** 2, 1.6 ** 2, num_frames)) * size.value

# Render frames in an embarrassingly parallel loop
for img_index, (azimuth, radius) in enumerate(
        zip(rotation_path, radial_path)
):

    # Parallelize across ranks
    # If image not allocated to specific rank, skip iteration
    if img_index % mpi_size != mpi_rank:
        continue

    camera_args = {'x': xCen.value,
                   'y': yCen.value,
                   'z': zCen.value,
                   'r': 10#radius,
                   't': 0,
                   'p': azimuth,
                   'zoom': 1,
                   'roll': 0,
                   'xsize': resolution,
                   'ysize': resolution,
                   'extent': [
                       - 0.8 * size.value,
                       0.8 * size.value,
                       - 0.8 * size.value,
                       0.8 * size.value
                   ]}

    density_map = make_image(data.gas, project="densities", camera_kwargs=camera_args) / rho_crit.value
    mass_map = make_image(data.gas, project="masses", camera_kwargs=camera_args)
    data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures
    mass_weighted_temp_map = make_image(data.gas, project="mass_weighted_temps", camera_kwargs=camera_args)
    temp_map = mass_weighted_temp_map / mass_map

    # Construct the 2 channels and merge map
    bin_temp = binary_normalise(
        np.log10(temp_map),
        bin_min=color_bounds_log10temperature[0],
        bin_max=color_bounds_log10temperature[1]
    )
    bin_dens = binary_normalise(
        np.log10(density_map),
        bin_min=color_bounds_log10density[0],
        bin_max=color_bounds_log10density[1]
    )
    combined_map = combine_layers(bin_dens, bin_temp, res=resolution)

    # Set-up figure and axes instance
    fig = plt.figure(figsize=(8, 8), dpi=resolution // 8)
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(0, 0, 1, 1)
    plt.axis('off')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])

    ax.imshow(combined_map, origin="lower", extent=region)

    # Create the legend
    legend_resolution = 100
    x_legend, y_legend = np.meshgrid(
        np.linspace(0, 1, legend_resolution),
        np.linspace(0, 1, legend_resolution)
    )
    cmap2d = combine_layers(y_legend, x_legend, res=legend_resolution)

    cax = fig.add_axes([0.87, 0.87, 0.1, 0.1])
    cax.imshow(cmap2d, origin="lower", extent=(color_bounds_log10temperature + color_bounds_log10density))
    force_aspect(cax)
    cax.set_xlabel(r"$\log_{10} (T / \rm{K})$")
    cax.set_ylabel(r"$\log_{10} (\rho / \rho_{\rm{crit}})$")
    cax.spines['bottom'].set_color('white')
    cax.spines['top'].set_color('white')
    cax.spines['right'].set_color('white')
    cax.spines['left'].set_color('white')
    cax.xaxis.label.set_color('white')
    cax.yaxis.label.set_color('white')
    cax.tick_params(axis='x', colors='white')
    cax.tick_params(axis='y', colors='white')
    fig.savefig(f'out_{img_index:04d}.png')

# Compose video from frames at the end of the rendering
# ffmpeg -framerate 60 -i out_%04d.png -vf scale=-2:480,format=yuv420p -c:v libx264 -profile:v high -c:a aac -strict experimental -b:a 192k -movflags faststart -crf 18 out.mp4