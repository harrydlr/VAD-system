import os
import json
import torch
import torchaudio
from pathlib import Path
from loguru import logger
from speechbrain.pretrained import VAD as SpeechBrainVAD

class VADLabeler:
    def __init__(self, model_dir, labels_dir):
        self.model = self.load_vad_model(model_dir)
        self.labels_dir = labels_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_vad_model(self, model_dir):
        model = SpeechBrainVAD.from_hparams(source=model_dir, savedir="pretrained_models/vad-crdnn-libriparty")
        model.to(self.device)
        model.eval()
        return model

    def label_audio_file(self, file_path):
        logger.info(f"Labeling file {file_path} ...")
        signal, sr = torchaudio.load(file_path)

        # Compute frame-level posteriors
        prob_chunks = self.model.get_speech_prob_file(file_path)

        # Apply a threshold on top of the posteriors
        prob_th = self.model.apply_threshold(prob_chunks).float()

        # Derive the candidate speech segments
        boundaries = self.model.get_boundaries(prob_th)

        # Apply energy VAD within each candidate speech segment (optional)
        boundaries = self.model.energy_VAD(file_path, boundaries)

        # Merge segments that are too close
        boundaries = self.model.merge_close_segments(boundaries, close_th=0.250)

        # Remove segments that are too short
        boundaries = self.model.remove_short_segments(boundaries, len_th=0.250)

        # Double-check speech segments (optional)
        boundaries = self.model.double_check_speech_segments(boundaries, file_path, speech_th=0.5)

        # Convert any PyTorch tensors to floats
        boundaries = [[float(start), float(end)] for start, end in boundaries]

        # Create the labels dictionary
        labels = {"speech_segments": []}
        for start, end in boundaries:
            labels["speech_segments"].append({"start_time": start, "end_time": end})

        # Record labels to .json
        base_name = Path(file_path).stem
        out_fn = f"{base_name}.json"
        out_fp = os.path.join(self.labels_dir, out_fn)
        with open(out_fp, "w") as f:
            json.dump(labels, f)  # No need to specify default for conversion

        nb_preds = len(labels["speech_segments"])
        logger.info(f"{nb_preds} predictions recorded to {out_fp}")
