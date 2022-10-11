#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 09:07:06 2022

@author: josephbriggs
"""
import random
import glob
import pathlib
import itertools
from PIL import Image, ImageFont, ImageDraw, ImageFilter
import torchvision.transforms as transforms


# font = ImageFont.truetype("Arial-Bold.ttf",14)

def generate_dataset(hr_output_directory, lr_output_directory, num):

    fonts = glob.glob("roman_fonts/*.ttf")

    with open('words.txt') as f:
        words = f.readlines()
    random.shuffle(words)

    for i, word in enumerate(itertools.cycle(words)):
        if i > num:
            break

        size = (1000, 1000)
        W, H = size
        background_color = random.randint(0, 255)
        image = Image.new('L', size, background_color)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(random.choice(fonts), 60)
        _, _, w, h = draw.textbbox((0, 0), word, font=font)
        draw.text((0, 0), word, font=font, fill=random.randint(0, 255))
        image = image.crop((0, 0, w, h))
        transform = transforms.RandomAffine((-180, 180), scale=(0.5, 2),
                                            shear=(-10, 10, -10, 10), fill=background_color)
        image = transform(image)
        transform = transforms.CenterCrop(50)
        image = transform(image)

        low_res_image = image.resize((25, 25))
        low_res_image = low_res_image.filter(
            ImageFilter.GaussianBlur(radius=1))

        # img.save("a_test.png")
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(image,cmap="gray")
        # ax[1].imshow(low_res_image,cmap="gray")
        image.save(f"{hr_output_directory}{str(i):0>7}.png")
        low_res_image.save(f"{lr_output_directory}{str(i):0>7}.png")
    print('Generated dataset')


def main():

    pathlib.Path('hr_text_train').mkdir(parents=True, exist_ok=True)
    pathlib.Path('lr_text_train').mkdir(parents=True, exist_ok=True)
    pathlib.Path('hr_text_test').mkdir(parents=True, exist_ok=True)
    pathlib.Path('lr_text_test').mkdir(parents=True, exist_ok=True)
    generate_dataset('hr_text_train/', 'lr_text_train/', 100000)
    generate_dataset('hr_text_test/', 'lr_text_test/', 10000)


if __name__ == "__main__":
    main()
