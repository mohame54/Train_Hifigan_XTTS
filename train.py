from argparse import ArgumentParser
import torch
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor

from datasets.preprocess import load_wav_feat_spk_data
from configs.gpt_hifigan_config import GPTHifiganConfig
from models.gpt_gan import GPTGAN

class GPTHifiganTrainer:
    def __init__(self, config):
        self.config = config
        # init audio processor
        self.ap = AudioProcessor(**config.audio.to_dict())
        # load training samples
        self.eval_samples, self.train_samples = load_wav_feat_spk_data(config.data_path, config.mel_path, config.spk_path, config.eval_split_size)
        self.model = GPTGAN(config, self.ap)

        if config.pretrain_path is not None:
            state_dict = torch.load(config.pretrain_path)
            hifigan_state_dict = {k.replace("xtts.hifigan_decoder.waveform_decoder.", ""): v for k, v in state_dict["model"].items() if "hifigan_decoder" in k and "speaker_encoder" not in k}
            self.model.model_g.load_state_dict(hifigan_state_dict, strict=False)

            if config.train_spk_encoder:
                speaker_encoder_state_dict = {k.replace("xtts.hifigan_decoder.speaker_encoder.", ""): v for k, v in state_dict["model"].items() if "hifigan_decoder" in k and "speaker_encoder" in k}
                self.model.speaker_encoder.load_state_dict(speaker_encoder_state_dict, strict=True)

    def train(self):
        # init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(), config, config.output_path, model=self.model, train_samples=self.train_samples, eval_samples=self.eval_samples
        )
        trainer.fit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=32, type=int)
    parser.add_argument("--sample_rate", default=24000, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--train_epochs", default=5, type=int)
    parser.add_argument("--test_epochs", default=1, type=int)
    parser.add_argument("--dataset_name", default="libritts", type=str)
    parser.add_argument("--mixed_pre", default=True, type=bool)
    args = parser.parse_args()
    config = GPTHifiganConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.val_batch_size,
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers,
        run_eval=True,
        test_delay_epochs=args.test_epochs,
        epochs=args.train_epochs,
        seq_len=8192,
        output_sample_rate=args.sample_rate,
        gpt_latent_dim = 1024,
        pad_short=2000,
        use_noise_augment=False,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=args.mixed_pre,
        lr_gen=1e-4,
        lr_disc=1e-4,
        use_stft_loss=True,
        use_l1_spec_loss=True,
        data_path=f"{args.dataset_path}/wavs",
        mel_path=f"{args.dataset_path}/gpt_latents",
        spk_path =f"{args.dataset_path}/speaker_embeddings",
        output_path="outputs",
        pretrain_path="XTTS-v2/model.pth",
        train_spk_encoder=False,
    )

    hifigan_trainer = GPTHifiganTrainer(config=config)
    hifigan_trainer.train()
