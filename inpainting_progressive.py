from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
import cv2
import numpy as np
import torch
from skimage import io

def inpainting(opt, it, n_it):
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
        if it == 0:
            real = functions.read_image(opt)
        else:
            input_name = '%d_start_scale=%d.png' % (it-1, opt.inpainting_start_scale)
            x = io.imread('%s/%s' % (dir2save, input_name))
            x = functions.np2torch(x, opt)
            real = x[:, 0:3, :, :]
        real = functions.adjust_scales2image(real, opt)
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        if (opt.inpainting_start_scale < 1) | (opt.inpainting_start_scale > (len(Gs)-1)):
            print("injection scale should be between 1 and %d" % (len(Gs)-1))
        else:
            if it == 0:
                # Average color
                m = cv2.imread('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]))
                m = 1 - m / 255
                img = cv2.imread('%s/%s' % (opt.input_dir, opt.ref_name))

                if not opt.multiple_holes:
                    positions = np.where(m == 0)
                    h_low = positions[0][0]
                    h_high = positions[0][-1]
                    w_low = positions[1][0]
                    w_high = positions[1][-1]
                else:
                    bboxes = np.load(opt.ref_dir + '/' + opt.ref_name[:-4] + '_bboxes.npy')
                    wlow = bboxes[:, 0]
                    hlow = bboxes[:, 1]
                    whigh = bboxes[:, 2]
                    hhigh = bboxes[:, 3]

                for j in range(3):
                    window = 10
                    if not opt.multiple_holes:
                        img[:, :, j][m[:, :, j] == 0] = img[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                                                        max(w_low - window, 0):min(w_high + window, m.shape[1]), j][
                            m[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                            max(w_low - window, 0):min(w_high + window, m.shape[1]), j] == 1].mean()
                    else:
                        # img[:, :, j][m[:, :, j] == 0] = img[:, :, j][m[:, :, j] == 1].mean()
                        for i in range(len(bboxes)):
                            positions = np.where(m[hlow[i]:hhigh[i], wlow[i]:whigh[i],:] == 0)
                            h_low = positions[0][0]
                            h_high = positions[0][-1]
                            w_low = positions[1][0]
                            w_high = positions[1][-1]
                            img[h_low:h_high, w_low:w_high, j][m[h_low:h_high, w_low:w_high, j] == 0] = img[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                                                            max(w_low - window, 0):min(w_high + window, m.shape[1]), j][
                                m[max(h_low - window, 0):min(h_high + window, m.shape[0]),
                                max(w_low - window, 0):min(w_high + window, m.shape[1]), j] == 1].mean()

                cv2.imwrite('%s/%s_averaged%s' % (dir2save, opt.input_name[:-4], opt.input_name[-4:]), img)

                ref = functions.read_image_dir('%s/%s_averaged%s' % (dir2save, opt.ref_name[:-4], opt.ref_name[-4:]), opt)
                mask = functions.read_image_dir('%s/%s_mask%s' % (opt.ref_dir,opt.ref_name[:-4],opt.ref_name[-4:]), opt)
            else:
                # Average color
                m = cv2.imread('%s/%s_mask_%d%s' % (opt.ref_dir, opt.ref_name[:-4], it-1, opt.ref_name[-4:]))
                m = 1 - m / 255
                img = cv2.imread('%s/%d_start_scale=%d%s' % (dir2save, it-1, opt.inpainting_start_scale, opt.ref_name[-4:]))
                for j in range(3):
                    img[:, :, j][m[:, :, j] == 0] = img[:, :, j][m[:, :, j] == 1].mean()
                cv2.imwrite('%s/%d_start_scale=%d_averaged%s' % (dir2save, it-1, opt.inpainting_start_scale, opt.ref_name[-4:]), img)

                ref = functions.read_image_dir('%s/%d_start_scale=%d_averaged%s' % (dir2save, it-1, opt.inpainting_start_scale, opt.ref_name[-4:]), opt)
                mask = functions.read_image_dir('%s/%s_mask_%d%s' % (opt.ref_dir, opt.ref_name[:-4], it-1, opt.ref_name[-4:]), opt)

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

            if it<n_it-1:
                # Discard changes for inner
                if not opt.multiple_holes:
                    positions = np.where(m == 0)
                    if positions[0][-1] > positions[0][0]:
                        h = min(positions[0][-1] - positions[0][0], 10)
                        m[positions[0][0]:positions[0][0] + h] = 1
                        m[positions[0][-1] - h+1:positions[0][-1]+1] = 1
                    if positions[1][-1] > positions[1][0]:
                        w = min(positions[1][-1] - positions[1][0], 10)
                        m[:, positions[1][0]:positions[1][0] + w] = 1
                        m[:, positions[1][-1] - w+1:positions[1][-1]+1] = 1
                    mt = torch.from_numpy(m)
                    mt = mt[:, :, :, None]
                    mt = mt.permute((3, 2, 0, 1))
                    for j in range(3):
                        mask[:,j,:,:][mt[:,j,:,:] == 0] = 0
                else:
                    bboxes = np.load(opt.ref_dir + '/' + opt.ref_name[:-4] + '_bboxes.npy')
                    wlow = bboxes[:, 0]
                    hlow = bboxes[:, 1]
                    whigh = bboxes[:, 2]
                    hhigh = bboxes[:, 3]
                    bm = np.copy(m)
                    for i in range(len(bboxes)):
                        positions = np.where(bm[hlow[i]:hhigh[i], wlow[i]:whigh[i], :] == 0)
                        if len(positions[0])>0:
                            if positions[0][-1] > positions[0][0]:
                                h = min(positions[0][-1] - positions[0][0], 10)
                                m[hlow[i]:hhigh[i], wlow[i]:whigh[i], :][positions[0][0]:positions[0][0] + h] = 1
                                m[hlow[i]:hhigh[i], wlow[i]:whigh[i], :][positions[0][-1] - h + 1:positions[0][-1] + 1] = 1
                            if positions[1][-1] > positions[1][0]:
                                w = min(positions[1][-1] - positions[1][0], 10)
                                m[hlow[i]:hhigh[i], wlow[i]:whigh[i], :][:, positions[1][0]:positions[1][0] + w] = 1
                                m[hlow[i]:hhigh[i], wlow[i]:whigh[i], :][:, positions[1][-1] - w + 1:positions[1][-1] + 1] = 1
                            mt = torch.from_numpy(m)
                            mt = mt[:, :, :, None]
                            mt = mt.permute((3, 2, 0, 1))
                            for j in range(3):
                                mask[:, j, hlow[i]:hhigh[i], wlow[i]:whigh[i]][mt[:, j, hlow[i]:hhigh[i], wlow[i]:whigh[i]] == 0] = 0
                                #mask[:, j, hlow[i]:hhigh[i], wlow[i]:whigh[i]][mt[:, j, hlow[i]:hhigh[i], wlow[i]:whigh[i]] == 1] = 1
            print(it)
            out = (1-mask)*real+mask*out

            """to_save = functions.convert_image_np(out.detach())

            for j in range(3):
                # out[:, j, :,:][mt[:, j, :,:] == 0] = 0
                #out[:, :, j][m[:, :, j] == 1].mean()
                to_save[m[:,:,j] == 0] = 0"""
            plt.imsave('%s/%d_start_scale=%d.png' % (dir2save, it, opt.inpainting_start_scale), functions.convert_image_np(out.detach()), vmin=0, vmax=1)
            m = (1 - m) * 255
            cv2.imwrite('%s/%s_mask_%d%s' % (opt.ref_dir, opt.ref_name[:-4], it, opt.ref_name[-4:]), m)

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='training image name', required=True)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    parser.add_argument('--inpainting_start_scale', help='inpainting injection scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='inpainting')
    parser.add_argument('--radius', help='radius harmonization', type=int, default=10)
    parser.add_argument('--multiple_holes', help='set true for multiple holes', action="store_true")
    parser.add_argument('--ref_name', help='training image name', type=str, default="")
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    if opt.ref_name == "":
        opt.ref_name = opt.input_name

    # Get size of the hole
    m = cv2.imread('%s/%s_mask%s' % (opt.ref_dir, opt.ref_name[:-4], opt.ref_name[-4:]))
    m = 1 - m / 255
    if not opt.multiple_holes:
        positions = np.where(m == 0)
        deltah = positions[0][-1] - positions[0][0]
        deltaw = positions[1][-1] - positions[1][0]
        n_it = min(deltah, deltaw) // 20 + 1  # 10 pixels at a time
    else:
        bboxes = np.load(opt.ref_dir + '/' + opt.ref_name[:-4] + '_bboxes.npy')
        wlow = bboxes[:, 0]
        hlow = bboxes[:, 1]
        whigh = bboxes[:, 2]
        hhigh = bboxes[:, 3]
        for i in range(len(bboxes)):
            positions = np.where(m[hlow[i]:hhigh[i], wlow[i]:whigh[i], :] == 0)
            if i==0:
                deltah = positions[0][-1] - positions[0][0]
                deltaw = positions[1][-1] - positions[1][0]
            else:
                if positions[0][-1] - positions[0][0]>deltah:
                    deltah = positions[0][-1] - positions[0][0]
                if positions[1][-1] - positions[1][0]>deltaw:
                    deltaw = positions[1][-1] - positions[1][0]
            h_low = positions[0][0]
            h_high = positions[0][-1]
            w_low = positions[1][0]
            w_high = positions[1][-1]
        n_it = min(deltah, deltaw) // 20 +1 # 10 pixels at a time
    # size =
    #np.square((m == 0).sum())
    print(n_it)

    for it in range(n_it):
        inpainting(opt, it, n_it)
        """"# update mask
        if positions[0][-1]> positions[0][0]:
            h = max(positions[0][-1] - positions[0][0],10)
            m[positions[0][0]:positions[0][0]+h] = 1
            m[positions[0][-1] - h:positions[0][-1]] = 1
        if positions[1][-1] > positions[1][0]:
            w = max(positions[1][-1] - positions[1][0],10)
            m[:,positions[1][0]:positions[1][0] + w] = 1
            m[:, positions[1][-1]-w:positions[1][-1]] = 1
        m = (1 - m)*255
        cv2.imwrite('%s/%s_mask_%d%s' % (opt.ref_dir, opt.input_name[:-4], it, opt.input_name[-4:]), m)
        m = 1 - m / 255
        positions = np.where(m == 0)"""
