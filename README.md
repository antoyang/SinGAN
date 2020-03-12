# SinGAN for Scalable Image And Video Inpainting

Extension of SinGAN (https://github.com/tamarott/SinGAN) on Image and Video Inpainting. 

Approach is summarized in the Project Report. It aims at being scalable and is mainly based on generalizing the training to damaged images with a loss masked on invalid pixels, then initializing with various schemes the hole and harmonizing it. 

For personalized image inpainting, I crop the part to be inpainted, create corresponding mask, train on the damaged image (use main_train.py --inpainting), and use inpainting.py. 

inpainting_progressive.py progressively recovers 10 pixel at the edges at a time, which is interesting for big holes. However, it might cause further harmonization problems at the multiple boarders. Another interesting possibility for big holes is to use generated samples as initialization.

I extended this approach on short videos, assuming distribution of patches does not change too much between frames and only training on the first frame. Video.ipynb (big file as I included many images as examples) demonstrates how to extract frames from a video, automatically crop out every instance of a given class in a given region and create a video from frames.  

Audio.ipynb demonstrates how to extract audio from a video, and to reverse the operation (integrate audio to a video).




