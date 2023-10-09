import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import torchaudio.transforms as T
import soundfile as sf
from extract_features import extract_audio_features
import torchaudio

class VADDataset(Dataset):
    def __init__(self, data_dir, label_dir, input_size, sample_rate):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.sample_rate = sample_rate
        self.file_list = self.load_file_list()
        self.audio_transform = None  # You can add your audio transformation here

    def load_file_list(self):
        # Load the list of audio files and labels from the data and label directories
        file_list = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    # Construct the corresponding label file path
                    label_file = os.path.join(
                        self.label_dir, os.path.splitext(file)[0] + ".json"
                    )
                    if os.path.exists(label_file):
                        file_list.append({"audio": os.path.join(root, file), "label": label_file})
        return file_list

    def load_audio_segment(self, file_path, start, end):
        # Load the entire audio waveform from the WAV file
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert start and end times to frame indices
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)

        # Slice the waveform to extract the desired segment
        audio_segment = waveform[:, start_frame:end_frame]

        # Check if the audio segment is shorter than the desired input size and pad if needed
        if audio_segment.shape[1] < self.input_size:
            padding = self.input_size - audio_segment.shape[1]
            audio_segment = torch.nn.functional.pad(audio_segment, (0, padding))

        return audio_segment

    def load_speech_segments(self, label_file):
        # Load speech segments from the JSON label file
        with open(label_file, "r") as json_file:
            label_data = json.load(json_file)
            speech_segments = label_data.get("speech_segments", [])

        return speech_segments

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_info = self.file_list[idx]

        # Load the audio segment
        audio_file = audio_info["audio"]
        label_file = audio_info["label"]

        # Load speech segments from the JSON label file
        speech_segments = self.load_speech_segments(label_file)

        # Iterate through speech segments and create entries for them
        entries = []
        sub_segment_id = 0
        for segment in speech_segments:
            start = int(segment["start_time"] * self.sample_rate)
            end = int(segment["end_time"] * self.sample_rate)

            # Load the sub-segment
            sub_segment = self.load_audio_segment(audio_file, start, end)
            sub_segment_len = sub_segment.shape[1]

            # For segments where speech is found, label them as 1
            label = 1

            entries.append({
                "file_id": os.path.basename(label_file).replace(".json", ""),
                "start": start,
                "end": end,
                "sub_segment": sub_segment,
                "sub_segment_id": sub_segment_id,
                "sub_segment_len": sub_segment_len,
                "label": label,
            })
            sub_segment_id += 1

        # Add entries for segments where no speech is found (label them as 0)
        audio_duration = torchaudio.info(audio_file).num_frames
        current_end = 0
        for entry in entries:
            start = entry["end"]
            end = entry["start"] if entry["start"] > current_end else current_end
            sub_segment = self.load_audio_segment(audio_file, start, end)
            sub_segment_len = sub_segment.shape[1]
            label = 0
            entries.append({
                "file_id": os.path.basename(label_file).replace(".json", ""),
                "start": start,
                "end": end,
                "sub_segment": sub_segment,
                "sub_segment_id": sub_segment_id,
                "sub_segment_len": sub_segment_len,
                "label": label,
            })
            sub_segment_id += 1
            current_end = entry["end"]

        return entries

def create_vad_dataset(data_dir, label_dir, input_size, output_dir):
    dataset = VADDataset(data_dir, label_dir, input_size, sample_rate=16000)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Print the attributes of the first few entries in the dataset
    num_entries_to_print = 5
    for idx in range(min(num_entries_to_print, len(dataset))):
        entries = dataset[idx]
        print(f"Entries for file {idx}:")
        for entry in entries:
            print(f"File ID: {entry['file_id']}")
            print(f"Start: {entry['start']}")
            print(f"End: {entry['end']}")
            print(f"Sub-segment ID: {entry['sub_segment_id']}")
            print(f"Sub-segment Length: {entry['sub_segment_len']}")
            print(f"Label: {entry['label']}")
            print("-" * 30)

if __name__ == "__main__":
    data_dir = "/home/harry/Downloads/test-clean/"
    label_dir = "/home/harry/Downloads/labels"
    input_size = 1024
    output_dir = "vad_datasets"

    create_vad_dataset(data_dir, label_dir, input_size, output_dir)
