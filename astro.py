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

resolution = 4096
snap_filepath_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/snapshots/L0300N0564_VR813_+1res_MinimumDistance_2749.hdf5"
velociraptor_properties_zoom = "/cosma6/data/dp004/dc-alta2/xl-zooms/hydro/L0300N0564_VR813_+1res_MinimumDistance/stf/L0300N0564_VR813_+1res_MinimumDistance_2749/L0300N0564_VR813_+1res_MinimumDistance_2749.properties"


def binary_normalise(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def gamma_correction(img: np.ndarray, gamma: float = 1.0):
    igamma = 1.0 / gamma
    imin, imax = img.min(), img.max()

    img_c = img.copy()
    img_c = ((img_c - imin) / (imax - imin)) ** igamma
    img_c = img_c * (imax - imin) + imin
    return img_c


def arcsinh_stretching(image_lrgb, beta: float = 0.1):

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
    combined_map[:, 3] = binary_normalise(combined_map[:, 3])
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

if not os.path.isfile("density_map.npy"):

    # Construct spatial mask to feed into swiftsimio
    mask = sw.mask(snap_filepath_zoom, spatial_only=False)
    region = [
        [xCen - size, xCen + size],
        [yCen - size, yCen + size],
        [zCen - 0.1 * size, zCen + 0.1 * size]
    ]
    mask.constrain_spatial(region)
    mask.constrain_mask("gas", "temperatures", 1.e5 * mask.units.temperature, 5.e9 * mask.units.temperature)
    data = sw.load(snap_filepath_zoom, mask=mask)
region = [
    xCen - 0.8 * size,
    xCen + 0.8 * size,
    yCen - 0.8 * size,
    yCen + 0.8 * size
]

if not os.path.isfile("density_map.npy"):
    density_map = project_gas(data, resolution=resolution, project="densities", parallel=True, region=region)
    np.save("density_map.npy", density_map)

density_map = np.load("density_map.npy")

# Calculate particle mass and rho_crit
data = sw.load(snap_filepath_zoom)
unitLength = data.metadata.units.length
unitMass = data.metadata.units.mass
rho_crit = unyt.unyt_quantity(
    data.metadata.cosmology_raw['Critical density [internal units]'],
    unitMass / unitLength ** 3
).to('Msun/Mpc**3')
density_map /= rho_crit.value

# density_map_rgb = plt.imshow(np.log10(density_map), origin="lower", extent=region, cmap="binary_r")

if not os.path.isfile("temp_map.npy"):
    mass_map = project_gas(data, resolution=resolution, project="masses", parallel=True, region=region)
    data.gas.mass_weighted_temps = data.gas.masses * data.gas.temperatures
    mass_weighted_temp_map = project_gas(data, resolution=resolution, project="mass_weighted_temps", parallel=True,
                                         region=region)
    temp_map = mass_weighted_temp_map / mass_map
    np.save("temp_map.npy", temp_map)

temp_map = np.load("temp_map.npy")
# temp_map_rgb = plt.imshow(np.log10(temp_map), origin="lower", extent=region, cmap="rainbow")

# Construct the 2 channels and merge map
bin_temp = binary_normalise(np.log10(temp_map))
bin_dens = binary_normalise(np.log10(density_map))
combined_map = bin_dens#combine_layers(bin_dens, bin_temp, res=resolution)

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
# legend_resolution = 100
# x_legend, y_legend = np.meshgrid(
#     np.linspace(0, 1, legend_resolution),
#     np.linspace(0, 1, legend_resolution)
# )
# cmap2d = combine_layers(y_legend, x_legend, res=legend_resolution)
#
# cax = fig.add_axes([0.87, 0.87, 0.1, 0.1])
# cax.imshow(cmap2d, origin="lower", extent=([
#     np.log10(np.min(temp_map)),
#     np.log10(np.max(temp_map)),
#     np.log10(np.min(density_map)),
#     np.log10(np.max(density_map))
# ]))
# force_aspect(cax)
# cax.set_xlabel(r"$\log_{10} (T / \rm{K})$")
# cax.set_ylabel(r"$\log_{10} (\rho / \rho_{\rm{crit}})$")
# cax.spines['bottom'].set_color('white')
# cax.spines['top'].set_color('white')
# cax.spines['right'].set_color('white')
# cax.spines['left'].set_color('white')
# cax.xaxis.label.set_color('white')
# cax.yaxis.label.set_color('white')
# cax.tick_params(axis='x', colors='white')
# cax.tick_params(axis='y', colors='white')
plt.show()
fig.savefig('out.png', bbox_inches='tight', pad_inches=0.)
