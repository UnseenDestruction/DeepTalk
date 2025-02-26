from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

def optimize_model(model):
    """Optimize model using TorchScript for CUDA 11.3 compatibility."""
    model.eval().cuda()
    traced_model = torch.jit.trace(model, torch.ones(1, 3, 256, 256).cuda())
    print("✅ Model optimized with TorchScript")
    return traced_model

def main(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"⚡ Using device: {device}")

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(os.path.split(sys.argv[0])[0], 'src/config'), args.size, args.old_version, args.preprocess)

    # 🔥 Load & Optimize Models
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    if device == "cuda":
        audio_to_coeff = optimize_model(audio_to_coeff)
        animate_from_coeff = optimize_model(animate_from_coeff)
    
    print("🖼️  Extracting 3DMM features from the source image...")
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size
    )
    if first_coeff_path is None:
        print("❌ Can't get the coeffs of the input image")
        return

    ref_eyeblink_coeff_path = preprocess_ref_video(args.ref_eyeblink, save_dir, preprocess_model, args.preprocess) if args.ref_eyeblink else None
    ref_pose_coeff_path = ref_eyeblink_coeff_path if args.ref_pose == args.ref_eyeblink else (
        preprocess_ref_video(args.ref_pose, save_dir, preprocess_model, args.preprocess) if args.ref_pose else None
    )

    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, args.pose_style, ref_pose_coeff_path)

    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # Generate final video
    data = get_facerender_data(
        coeff_path, crop_pic_path, first_coeff_path, audio_path, args.batch_size,
        args.input_yaw, args.input_pitch, args.input_roll, args.expression_scale, args.still, args.preprocess, args.size
    )
    
    result = animate_from_coeff.generate(
        data, save_dir, pic_path, crop_info, args.enhancer, args.background_enhancer, args.preprocess, args.size
    )

    shutil.move(result, f"{save_dir}.mp4")
    print(f"🎥 Generated video: {save_dir}.mp4")

    if not args.verbose:
        shutil.rmtree(save_dir)

def preprocess_ref_video(ref_video_path, save_dir, preprocess_model, preprocess):
    """Extract 3DMM features from a reference video."""
    videoname = os.path.splitext(os.path.split(ref_video_path)[-1])[0]
    ref_frame_dir = os.path.join(save_dir, videoname)
    os.makedirs(ref_frame_dir, exist_ok=True)
    print(f"📹 Extracting 3DMM from reference video: {videoname}")
    return preprocess_model.generate(ref_video_path, ref_frame_dir, preprocess, source_image_flag=False)[0]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="reference video for eye blinking")
    parser.add_argument("--ref_pose", default=None, help="reference video for pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="checkpoint path")
    parser.add_argument("--result_dir", default='./results', help="output path")
    parser.add_argument("--pose_style", type=int, default=0, help="pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size for facerender")
    parser.add_argument("--size", type=int, default=256, help="image size for facerender")
    parser.add_argument("--expression_scale", type=float, default=1., help="expression intensity")
    parser.add_argument("--input_yaw", nargs='+', type=int, default=None, help="yaw degree")
    parser.add_argument("--input_pitch", nargs='+', type=int, default=None, help="pitch degree")
    parser.add_argument("--input_roll", nargs='+', type=int, default=None, help="roll degree")
    parser.add_argument("--enhancer", type=str, default=None, help="Face enhancer (gfpgan, RestoreFormer)")
    parser.add_argument("--background_enhancer", type=str, default=None, help="Background enhancer (realesrgan)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--face3dvis", action="store_true", help="Generate 3D face and landmarks")
    parser.add_argument("--still", action="store_true", help="Keep the original crop for full-body animation")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="Image preprocessing method")
    parser.add_argument("--verbose", action="store_true", help="Save intermediate outputs")
    parser.add_argument("--old_version", action="store_true", help="Use old pth model instead of safetensor")

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)
