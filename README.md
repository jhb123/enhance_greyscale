# enhance_greyscale
neural net implemented with pytorch for upscaling greyscale images.

## Usage
To upscale an image, use:
```
python3 upscale_image.py -i "low_res_image.png" -o "high_Res_image.png"
```

To generate a set of training data from images within a file, use:
```
python generate_training_data.py -i "img_directory/" -o "gs_imgs" -r 4
```
The `-i` option is the directory containing the images that the network will be trained with. The `-o` option is the prefix of the directories with the training data in them. The `-r` option is the upscaling factor that is to be trained for.

To train the network,
```
python train_network.py -hr "gs_imgs_hr_x4" -lr "gs_imgs_lr_x4" -r 4 -a 1e-4 -b 10 -e 1 -d 10000
```
1. `-hr` is the directory containing high res images.
2. `-lr` is the directory containing low res images.
3. `-r` is upscaling factor.
4. `-a` is the learning rate.
5. `-b` is the batch size.
6. `-e` is the number of epochs.
7. `-d` is the number of training examples to be used.

## Example

![example_image](https://github.com/jhb123/enhance_greyscale/blob/main/example.png)
