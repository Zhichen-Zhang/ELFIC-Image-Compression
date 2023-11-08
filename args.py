import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ELFIC')
    parser.add_argument('--search_space', default='./SupernetSpace/', type=str,
                        help='Search space of currenting model')

    parser.add_argument("--description", type=str, default="Model is original")
    parser.add_argument("--CompressorName", type=str, default="ELFIC")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--state", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--search", type=bool, default=False)

    # model path
    parser.add_argument("--model_restore_path", type=str,
                        default="./checkpoints/checkpoint_latest.pth")  # load model weights path
    # for resume training or the second stage training.
    parser.add_argument("--load_pretrain", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--log_root", type=str, default="./logs")

    parser.add_argument("--mode_type", type=str, default='PSNR')
    # parser.add_argument("--rate_list", type=list, default=[0.0018, 0.0035, 0.0067, 0.013, 0.025])
    parser.add_argument("--rate_list", type=list, default=[0.020, 0.0383, 0.0566, 0.0749, 0.0932])

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--search_batch_size", type=int, default=64)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--image_size", type=list, default=[256, 256, 3])

    # Dataset preprocess parameters
    parser.add_argument("--dataset_root", type=str, default='./data/vimeo_interp_train')
    parser.add_argument("--dataset_root_a", type=str, default='./data/flicker_2W_images')
    parser.add_argument("--search_root", type=str, default='./CLIC_Search')
    parser.add_argument("--valid_root", type=str, default='./valid')

    parser.add_argument("--frames", type=int, default=7)

    # Optimizer parameters
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aux_lr', type=float, default=1e-3)
    parser.add_argument("--clip_max_norm", default=1, type=float, help="gradient clipping max norm ")

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=2, help='# num_workers')
    return parser.parse_args()

