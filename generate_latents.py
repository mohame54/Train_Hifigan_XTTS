import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchaudio
import torch
from torch.utils.data import DataLoader
from pydub import AudioSegment
from argparse import ArgumentParser
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig, XttsAudioConfig
from TTS.tts.models.xtts import load_audio

from models.gpt_decode import GPTDecode
from datasets.dataset_xtts import GPTXTTSDataset


def custom_formatter(root_path, meta, **kwargs):  # pylint: disable=unused-argument
    wavs_dir = os.path.join(root_path, "wavs")
    meta_df = pd.read_csv(os.path.join(root_path, "metadata.csv"))
    items = []
    for i in range(len(meta_df)):
        row = meta_df.iloc[i]
        wav_path = os.path.join(wavs_dir, row['wav_path'])
        text = row['text']
        sp_id = row['speaker_id']
        items.append({
            "text":text,
            "audio_file":wav_path,
            "speaker_name":sp_id,
            "root_path": root_path})
    return items



class GPTDecoder:
    def __init__(self, config, config_dataset):
        self.config = config
        self.config_dataset = config_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_samples, _ = load_tts_samples(
            config_dataset, formatter=custom_formatter,
        )
        self.tokenizer = VoiceBpeTokenizer(config.model_args.tokenizer_file)
        self.dataset = GPTXTTSDataset(config, self.train_samples, self.tokenizer, config.audio.sample_rate, is_eval=True)
        self.loader = DataLoader(self.dataset, collate_fn=self.dataset.collate_fn, batch_size=self.config.batch_size)
        self.model = GPTDecode.init_from_config(config).to(self.device)
    
    def load_audio_16k(self, files):
        audios = []
        for file in files:
            audio = load_audio(file, self.config.audio.sample_rate).to(self.device)
            audio = audio[:, : self.config.audio.sample_rate * 30]

            audio_16k = torchaudio.functional.resample(audio, self.config.audio.sample_rate, 16000).squeeze(0)
            audios.append(audio_16k)

        max_len   = max([_.size(0) for _ in audios])
        audio_padded  = torch.zeros(len(audios), max_len)
        for i in range(len(audios)):
            audio_padded[i, : audios[i].size(0)] = audios[i]

        return audio_padded

    def generate(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "gpt_latents"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "speaker_embeddings"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "synthesis"), exist_ok=True)

        for id, batch in enumerate(tqdm(self.loader)):
            batch["text_lengths"] = batch["text_lengths"].to(self.device)
            batch["wav_lengths"] = batch["wav_lengths"].to(self.device)
            batch["cond_idxs"] = batch["cond_idxs"].to(self.device)
            batch["wav"] = batch["wav"].to(self.device)

            batch = self.model.format_batch_on_device(batch)

            cond_mels = batch["cond_mels"].to(self.device)
            text_inputs = batch["text_inputs"].to(self.device)
            text_lengths = batch["text_lengths"].to(self.device)
            audio_codes = batch["audio_codes"].to(self.device)
            wav_lengths = batch["wav_lengths"].to(self.device)
            cond_idxs = batch["cond_idxs"].to(self.device)
            cond_lens = batch["cond_lens"]
            code_lengths = torch.ceil(wav_lengths / self.model.xtts.gpt.code_stride_len).long()
            self.model.eval()
            audio_16k = self.load_audio_16k(batch["filenames"]).to(self.device)
            with torch.no_grad():
              with torch.autocast("cuda", dtype=torch.float16):
                speaker_embedding = self.model.xtts.hifigan_decoder.speaker_encoder.forward(audio_16k, l2_norm=True).unsqueeze(-1)

                latents = self.model.generate(
                    text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
                )
                wav = []
                for i in range(self.config.batch_size):
                    wav.append(self.model.xtts.hifigan_decoder(latents[i][: code_lengths[i]].unsqueeze(0), g=speaker_embedding[i]).detach().cpu().squeeze().to(torch.float32))
            latents = latents.to(torch.float32)

            for i in range(self.config.batch_size):
                file_name = batch["filenames"][i].split("/")[-1]

                raw_audio = AudioSegment.from_file(batch["filenames"][i])
                raw_audio = raw_audio.set_frame_rate(self.config.audio.output_sample_rate)
                raw_audio.export(os.path.join(output_dir, "wavs", file_name), format="wav")
                torchaudio.save(os.path.join(output_dir, "synthesis", file_name), torch.tensor(wav[i]).unsqueeze(0), self.config.audio.output_sample_rate)

                with open(os.path.join(output_dir, "gpt_latents", file_name.replace(".wav", ".npy")), "wb") as f:
                    np.save(f, latents[i][: code_lengths[i]].detach().squeeze(0).transpose(0, 1).cpu())
                
                with open(os.path.join(output_dir, "speaker_embeddings", file_name.replace(".wav", ".npy")), "wb") as f:
                    np.save(f, speaker_embedding[i].detach().squeeze(0).squeeze(1).cpu())




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--lang",default="en-us", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--sample_rate", default=24000, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--dataset_name", default="libritts", type=str)
    args = parser.parse_args()

    audio_config = XttsAudioConfig(sample_rate=args.sample_rate, dvae_sample_rate=22050, output_sample_rate=24000)
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file="XTTS-v2/mel_stats.pth",
        dvae_checkpoint="XTTS-v2/dvae.pth",
        xtts_checkpoint="XTTS-v2/model.pth",
        tokenizer_file="XTTS-v2/vocab.json",
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    config = GPTTrainerConfig(
        audio=audio_config,
        model_args=model_args,
        batch_size = args.batch_size,
        num_loader_workers=args.num_workers,
    )

    dataset_en = BaseDatasetConfig(
        formatter=custom_formatter,
        dataset_name=args.dataset_name,
        path=args.dataset_path,
        language=args.lang,
    )
    dataset_config = [dataset_en]

    gpt_decode = GPTDecoder(config, dataset_config)
    gpt_decode.generate(output_dir=args.dataset_name + "_latents")
