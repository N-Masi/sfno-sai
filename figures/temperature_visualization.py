import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pdb

# Create a figure and a single subplot with PlateCarree projection
fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': ccrs.PlateCarree()})

# Data for plotting
lats = np.linspace(-90, 90, 192)
lons = np.linspace(-180, 180, 288)
lon_grid, lat_grid = np.meshgrid(lons, lats)
values_pred = np.random.randn(192, 288)
values_sim = np.random.randn(192, 288)

for ax in axes:
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    gridlines = ax.gridlines(draw_labels=True)
    gridlines.top_labels = False
    gridlines.right_labels = False
gridlines.left_labels = False

pcolormesh = axes[0].pcolormesh(
    lon_grid, lat_grid, values_pred, cmap='coolwarm', shading='auto',
    transform=ccrs.PlateCarree()
)
axes[0].set_title('Predicted Temperature After 35 Year Rollout', fontsize=14)

pcolormesh = axes[1].pcolormesh(
    lon_grid, lat_grid, values_sim, cmap='coolwarm', shading='auto',
    transform=ccrs.PlateCarree()
)
axes[1].set_title('Ground-Truth Temperature After 35 Year Simulation', fontsize=14)

# Add a colorbar
cbar = fig.colorbar(pcolormesh, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
cbar.set_label('°C')

# Show the plot
plt.savefig("figures/nv.png")
