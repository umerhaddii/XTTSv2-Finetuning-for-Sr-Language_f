import os
import gc
import traceback

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

def train_gpt():
    try:
        # Hardcoded parameters
        output_path = "/kaggle/working/checkpoints/"
        train_csv = "/kaggle/input/serbian-xtts/metadata_train.csv"
        eval_csv = "/kaggle/input/serbian-xtts/metadata_eval.csv"
        language = "sr"
        num_epochs = 10
        batch_size = 4
        grad_acumm = 2
        max_audio_length = 330750
        max_text_length = 400
        lr = 5e-6
        weight_decay = 1e-2
        save_step = 500

        print("=" * 50)
        print("XTTS Serbian Fine-tuning Started")
        print("=" * 50)

        # Logging parameters
        RUN_NAME = "GPT_XTTS_FT_Serbian"
        PROJECT_NAME = "XTTS_trainer"
        DASHBOARD_LOGGER = "tensorboard"
        LOGGER_URI = None
        OUT_PATH = output_path

        # Training Parameters
        OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
        START_WITH_EVAL = False
        BATCH_SIZE = batch_size
        GRAD_ACUMM_STEPS = grad_acumm

        print("\n[1/6] Configuring dataset...")
        try:
            config_dataset = BaseDatasetConfig(
                formatter="serbian_formatter",
                dataset_name="serbian_voice",
                path="/kaggle/input/serbian-xtts",
                meta_file_train="metadata_train.csv",
                meta_file_val="metadata_eval.csv",
                language=language,
            )
            DATASETS_CONFIG_LIST = [config_dataset]
            print("✓ Dataset configured")
        except Exception as e:
            print(f"✗ Dataset configuration failed: {str(e)}")
            traceback.print_exc()
            raise

        print("\n[2/6] Setting up checkpoint paths...")
        try:
            CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
            os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

            # DVAE files
            DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
            MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
            DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
            MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")

            # XTTS files
            TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
            XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
            XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
            TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
            XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
            XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "config.json")

            # Download if needed
            if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
                print("  Downloading DVAE files...")
                ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
            
            if not os.path.isfile(TOKENIZER_FILE):
                print("  Downloading tokenizer...")
                ModelManager._download_model_files([TOKENIZER_FILE_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
            
            if not os.path.isfile(XTTS_CHECKPOINT):
                print("  Downloading XTTS checkpoint...")
                ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
            
            if not os.path.isfile(XTTS_CONFIG_FILE):
                print("  Downloading XTTS config...")
                ModelManager._download_model_files([XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)
            
            print("✓ Checkpoints ready")
        except Exception as e:
            print(f"✗ Checkpoint setup failed: {str(e)}")
            traceback.print_exc()
            raise

        print("\n[3/6] Initializing model configuration...")
        try:
            model_args = GPTArgs(
                max_conditioning_length=132300,
                min_conditioning_length=11025,
                debug_loading_failures=False,
                max_wav_length=max_audio_length,
                max_text_length=max_text_length,
                mel_norm_file=MEL_NORM_FILE,
                dvae_checkpoint=DVAE_CHECKPOINT,
                xtts_checkpoint=XTTS_CHECKPOINT,
                tokenizer_file=TOKENIZER_FILE,
                gpt_num_audio_tokens=1026,
                gpt_start_audio_token=1024,
                gpt_stop_audio_token=1025,
                gpt_use_masking_gt_prompt_approach=True,
                gpt_use_perceiver_resampler=True,
            )

            audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
            
            config = GPTTrainerConfig()
            config.load_json(XTTS_CONFIG_FILE)
            config.epochs = num_epochs
            config.output_path = OUT_PATH
            config.model_args = model_args
            config.run_name = RUN_NAME
            config.project_name = PROJECT_NAME
            config.run_description = "GPT XTTS Serbian fine-tuning"
            config.dashboard_logger = DASHBOARD_LOGGER
            config.logger_uri = LOGGER_URI
            config.audio = audio_config
            config.batch_size = BATCH_SIZE
            config.num_loader_workers = 8
            config.eval_split_max_size = 256
            config.print_step = 50
            config.plot_step = 100
            config.log_model_step = 100
            config.save_step = save_step
            config.save_n_checkpoints = 1
            config.save_checkpoints = True
            config.print_eval = False
            config.optimizer = "AdamW"
            config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
            config.optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": weight_decay}
            config.lr = lr
            config.lr_scheduler = "MultiStepLR"
            config.lr_scheduler_params = {"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}
            config.test_sentences = []
            
            print("✓ Model configuration loaded")
        except Exception as e:
            print(f"✗ Model configuration failed: {str(e)}")
            traceback.print_exc()
            raise

        print("\n[4/6] Initializing model...")
        try:
            model = GPTTrainer.init_from_config(config)
            print("✓ Model initialized")
        except Exception as e:
            print(f"✗ Model initialization failed: {str(e)}")
            traceback.print_exc()
            raise

        print("\n[5/6] Loading training samples...")
        try:
            train_samples, eval_samples = load_tts_samples(
                DATASETS_CONFIG_LIST,
                eval_split=True,
                eval_split_max_size=config.eval_split_max_size,
                eval_split_size=config.eval_split_size,
            )
            print(f"✓ Loaded {len(train_samples)} train samples, {len(eval_samples)} eval samples")
        except Exception as e:
            print(f"✗ Sample loading failed: {str(e)}")
            traceback.print_exc()
            raise

        print("\n[6/6] Starting training...")
        try:
            trainer = Trainer(
                TrainerArgs(
                    restore_path=None,
                    skip_train_epoch=False,
                    start_with_eval=START_WITH_EVAL,
                    grad_accum_steps=GRAD_ACUMM_STEPS
                ),
                config,
                output_path=os.path.join(output_path, "run", "training"),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
            )
            trainer.fit()
            
            trainer_out_path = trainer.output_path
            print(f"\n✓ Training complete! Checkpoints saved to: {trainer_out_path}")
            
            # Cleanup
            del model, trainer, train_samples, eval_samples
            gc.collect()
            
            return trainer_out_path
        except Exception as e:
            print(f"✗ Training failed: {str(e)}")
            traceback.print_exc()
            raise

    except Exception as e:
        print("\n" + "=" * 50)
        print("FATAL ERROR")
        print("=" * 50)
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n🚀 XTTS v2 Serbian Fine-tuning")
    print("Hardcoded for Kaggle environment\n")
    
    result = train_gpt()
    
    if result:
        print(f"\n✅ SUCCESS! Model saved to: {result}")
    else:
        print("\n❌ FAILED! Check errors above.")
