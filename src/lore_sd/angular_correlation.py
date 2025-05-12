# This file will take two files with ODFs as input and write the angular correlation coefficient (Anderson, 2005) to a file.

import argparse

import numpy as np
import sys

# sys.path.append('/DIRECTORY_TO_REPO/LoRE_SD/LoRE-SD/src')
# sys.path.append('/DIRECTORY_TO_REPO/LoRE_SD/LoRE-SD')

from lore_sd.mrtrix_io.io import load_mrtrix, save_mrtrix
from lore_sd.mrtrix_io.io.image import Image
from lore_sd.utils import SphericalHarmonics as sh

def main():
    parser = argparse.ArgumentParser(description='Calculate the angular correlation coefficient between two ODFs.')
    parser.add_argument('odf1', type=str, help='The first ODF file.')
    parser.add_argument('odf2', type=str, help='The second ODF file.')
    parser.add_argument('output', type=str, help='The output file.')

    args = parser.parse_args()

    data1 = load_mrtrix(args.odf1)
    data2 = load_mrtrix(args.odf2)

    # Check if the last dimension is the same
    if data1.shape[-1] != data2.shape[-1]:
        raise ValueError('The last dimension of the two input files must be the same. Got {} and {}.'.format(data1.shape[-1], data2.shape[-1]))

    # Calculate the angular correlation coefficient
    angular_correlation = sh.angularCorrelation(data1.data, data2.data)

    # Save the result
    save_mrtrix(args.output, Image(angular_correlation, vox=data1.vox,
                                   comments=[f'Angular correlation coefficient between '
                                             f'{args.odf1.split("/")[-1]} and {args.odf2.split("/")[-1]}']))

if __name__ == '__main__':
    main()
