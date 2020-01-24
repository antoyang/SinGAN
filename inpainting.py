from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import torch
import cv2

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    parser.add_argument('--radius', help='radius harmonization', type=int, default = 10)
    parser.add_argument('--multiple_holes', help = 'set true for multiple holes', action = "store_true")
    parser.add_argument('--ref_name', help='training image name', type = str, default = "")
    """parser.add_argument('--hl', default=0, type=int)
    parser.add_argument('--hh', default=0, type=int)
    parser.add_argument('--wl', default=0, type=int)
    parser.add_argument('--wh', default=0, type=int)"""
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    if opt.ref_name =="":
        opt.ref_name = opt.input_name
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    #elif (os.path.exists(dir2save)):
    #    print("output already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            #Average color
            m = cv2.imread('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]))
            m = 1 - m / 255
            img = cv2.imread('%s/%s' % (opt.input_dir, opt.ref_name))

            if not opt.multiple_holes:
                positions = np.where(m == 0)
                h_low = positions[0][0]
                h_high = positions[0][-1]
                w_low = positions[1][0]
                w_high = positions[1][-1]

            for j in range(3):
                window = 10
                if not opt.multiple_holes:
                    img[:, :, j][m[:, :, j] == 0] = img[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                                                    max(w_low - window, 0):min(w_high + window, m.shape[1]), j][
                        m[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                        max(w_low - window, 0):min(w_high + window, m.shape[1]), j] == 1].mean()
                else:
                    img[:, :, j][m[:, :, j] == 0] = img[:,:, j][m[:,:, j] == 1].mean()

            cv2.imwrite('%s/%s_averaged%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), img)

            ref = functions.read_image_dir('%s/%s_averaged%s' % (opt.input_dir, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
            mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)

            if ref.shape[3] != real.shape[3]:
                mask = imresize_to_shape(mask, [real.shape[2], real.shape[3]], opt)
                mask = mask[:, :, :real.shape[2], :real.shape[3]]
                ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
                ref = ref[:, :, :real.shape[2], :real.shape[3]]
            mask = functions.dilate_mask(mask, opt)

            N = len(reals) - 1
            n = opt.inpainting_start_scale
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
            out = SinGAN_generate(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            out = (1-mask)*real+mask*out
            plt.imsave('%s/start_scale=%d.png' % (dir2save,opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)