import xarray as xr
import s3fs

# Set up the S3 file system with anonymous access
s3 = s3fs.S3FileSystem(anon=True)

# Specify the S3 path
s3_path = 's3://ncar-cesm2-arise/ARISE-SAI-1.5-2045/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.001/atm/proc/tseries/hour_3/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DELAYED-2045.001.cam.h2.CAPE.2045010100-2045123100.nc'

try:
    # Open the dataset without loading it into memory
    with s3.open(s3_path, 'rb') as f:
        ds = xr.open_dataset(f, engine='netcdf4')

    # Print basic information about the dataset
    print(ds.info())

    # Print the coordinates
    print("\nCoordinates:")
    print(ds.coords)

    # Print the data variables
    print("\nData Variables:")
    print(ds.data_vars)

    # Print some attributes
    print("\nGlobal Attributes:")
    for attr in ds.attrs:
        print(f"{attr}: {ds.attrs[attr]}")

    # Close the dataset
    ds.close()

except Exception as e:
    print(f"An error occurred: {e}")