import audio_labeller_vf
import os

if __name__ == "__main__":
    data_dir = '/home/harry/Downloads/test-clean/'
    model_dir = 'speechbrain/vad-crdnn-libriparty'
    visualize = False  # Set to True if you want to visualize predictions
    labels_dir = '/home/harry/Downloads/labels'  # Modify this path as needed

    labeler = audio_labeller_vf.VADLabeler(model_dir, labels_dir)

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                labeler.label_audio_file(file_path)
