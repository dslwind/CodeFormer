import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs', 
            help='Input image, video or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
            help='Output folder. Default: results/<input_name>_<w>')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
            help='Balance the quality and fidelity. Default: 0.5')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight
    input_video = False
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_img_{w}'
    elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
        video_name = os.path.basename(args.input_path)[:-4]
        result_root = f'results/{video_name}_{w}'
        input_video = True
        vidreader.close()
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}_{w}'

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        input_img = cv2.imread(img_path)

        input_img = img2tensor(input_img / 255., bgr2rgb=True, float32=True)
        normalize(input_img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        input_img = input_img.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                # w is fixed to 0 since we didn't train the Stage III for colorization
                output_face = net(input_img, w=0, adain=True)[0] 
                restored_img = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
            del output_face
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_img = tensor2img(input_img, rgb2bgr=True, min_max=(-1, 1))

        restored_img = restored_img.astype('uint8')

        # save restored img
        if not args.has_aligned and restored_img is not None:
            if args.suffix is not None:
                basename = f'{basename}_{args.suffix}'
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)

    # save enhanced video
    if input_video:
        print('Video Saving...')
        img_path = os.path.join(result_root, 'final_results', '*.png')
        if args.suffix is not None:
            video_name = f'{video_name}_{args.suffix}.png'
        save_restore_path = os.path.join(result_root, f'{video_name}.mp4')
        
        import ffmpeg
        (
            ffmpeg
            .input(img_path, pattern_type='glob', framerate=fps)
            .output(save_restore_path, pix_fmt='yuv420p')
            .run()
        )

    print(f'\nAll results are saved in {result_root}')
