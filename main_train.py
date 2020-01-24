from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    # parser.add_argument('--hl', default = 0, type = int)
    # parser.add_argument('--hh', default = 0, type = int)
    # parser.add_argument('--wl', default = 0, type = int)
    # parser.add_argument('--wh', default = 0, type = int)
    parser.add_argument('--inpainting', action = 'store_true', help = 'store true to train for inpainting')
    parser.add_argument('--partial', action='store_true', help='store true to use partial convolutions')
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Inpainting')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        print(opt.stop_scale)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
