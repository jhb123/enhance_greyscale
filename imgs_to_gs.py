
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:56:55 2022

@author: josephbriggs
"""
import pathlib
import argparse
import cv2


def main():
    '''
    Converts files to greyscale.
    '''

    parser = argparse.ArgumentParser(description='Convert files to greyscale.')
    parser.add_argument('--input_path', "-i", type=str,
                        help='path to the image or directory of images. \
                If converting a directory, use *')
    parser.add_argument('--output_path', "-o", type=str,
                        help='output path where images will be saved.')
    parser.add_argument('--res', "-r", type=int,
                        help='downscale factor.')

    args = parser.parse_args()

    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # files = pathlib.Path(args.input_path).glob(r'/*.png|')
    file_extentions = ['png', 'jpeg', 'jpg']
    files = []
    for file_extension in file_extentions:
        files += pathlib.Path(args.input_path).glob(fr'*.{file_extension}')

    for file in files:
        file_name = file.name
        image = cv2.imread(str(file))
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        file_name_to_save = args.output_path+"/"+file_name
        print(file_name_to_save)
        cv2.imwrite(file_name_to_save, image_gs)

    print('converted files to greyscale')


if __name__ == "__main__":

    main()
