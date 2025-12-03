# download_checkpoint.py - Updated for current HF
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from huggingface_hub import hf_hub_download
import os

@dataclass
class DownloadArgs:
    output_path: str = field(
        default="checkpoints",
        metadata={"help": "Path to pretrained + checkpoint model"}
    )

def download(output_path: str = "checkpoints"):
    CHECKPOINTS_OUT_PATH = os.path.join(output_path, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
    
    repo_id = "coqui/XTTS-v2"
    files = [
        "dvae.pth",
        "mel_stats.pth",
        "vocab.json",
        "model.pth",
        "config.json",
        "speakers_xtts.pth"
    ]
    
    for file in files:
        file_path = os.path.join(CHECKPOINTS_OUT_PATH, file)
        if not os.path.isfile(file_path):
            print(f" > Downloading {file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=CHECKPOINTS_OUT_PATH,
                local_dir_use_symlinks=False
            )
    
    print(" > All files downloaded successfully!")

if __name__ == "__main__":
    parser = HfArgumentParser(DownloadArgs)
    args = parser.parse_args()
    download(output_path=args.output_path)
