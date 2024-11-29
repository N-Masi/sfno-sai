# # Create a figure and axes
# fig, axes = plt.subplots(1, 2, figsize=(14, 7))
# # Create a Basemap instance for the North Pole and South Pole
# m = Basemap(projection='npstere', boundinglat=10, lon_0=270, resolution='l', ax=axes[0])
# m_south = Basemap(projection='spstere', boundinglat=-10, lon_0=90, resolution='l', ax=axes[1])
# # Draw coastlines and countries on the North Pole map
# m.drawcoastlines()
# # m.drawcountries()  # Uncomment this if you want to draw country borders
# # Plot your variable (example: scatter plot)
# lats = np.linspace()
# lons = [0, 45, 90, -45, -90]
# values = [10, 20, 30, 40, 50]
# # Convert the lat/lon to map projection coordinates
# x, y = m(lons, lats)
# m.scatter(x, y, s=values, c=values, cmap='viridis', marker='o')

# # Draw coastlines on the South Pole map
# m_south.drawcoastlines()
# m_south.scatter(x, y, s=values, c=values, cmap='viridis', marker='o')

# # Set titles for the subplots
# axes[0].set_title('North Pole')
# axes[1].set_title('South Pole')

# # Show the plot
# plt.tight_layout()
# plt.show()

