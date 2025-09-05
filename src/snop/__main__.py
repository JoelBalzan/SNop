import argparse
import os
import sys

from .core.gen_ds import *




def main():
    parser = argparse.ArgumentParser(description="Optimise S/N dynamic spectrum from X Y polarisation data")
    parser.add_argument(
        "-f", "--frbname",
        type=str,
        default="FRB",
        help="Name of the FRB (default: FRB)"
    )
    parser.add_argument(
        "-x", "--xdata",
        type=str,
        required=True,
        help="Path to x polarisation data file"
    )
    parser.add_argument(
        "-y", "--ydata",
        type=str,
        required=True,
        help="Path to y polarisation data file"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default=os.getcwd(),
        help="Output directory (default: current working directory)"
    )
    parser.add_argument(
        "-d", "--dm",
        type=float,
        default=0.0,
        help="DM to dedisperse the data (default: 0.0)"
    )
    parser.add_argument(
        "-n", "--nchan",
        type=int,
        default=336,
        help="Number of frequency channels (default: 1)"
    )
    args = parser.parse_args()


    xDS, yDS = voltage_ds(args.frbname, args.xdata, args.ydata, args.dm, args.nchan, args.outdir)
    stokes_DS = gen_stokes_ds(args.frbname, xDS, yDS, args.dm, args.nchan, args.outdir)
    zoom_cube, meta = find_frb_and_zoom(stokes_DS, save_path=args.outdir)
    
    print(f"Zoomed cube shape: {zoom_cube.shape}")
    print(f"Metadata: {meta}")

if __name__ == "__main__":
    main()