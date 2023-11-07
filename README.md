# VAD-system
This is a 2 days experience trying to build a Voice Activity Detection system (VAD).

It is inspired by this work https://github.com/filippogiruzzi/voice_activity_detection/tree/master done in tensorflow and using the author's own model. The implementation in my repository is made in pytorch and uses the huggingface model Voice Activity Detection with a (small) CRDNN model trained on Libriparty (https://huggingface.co/speechbrain/vad-crdnn-libriparty) to create a tagged dataset to train the model with.

The audio data comes from the LibriSpeech ASR corpus (https://openslr.org/12/).

We need to run create_labels.py to create a folder with .json files containing the audio segments where speech is detected.

We will use data_set_creator.py to create the dataset that we will use to train, validate and test our model.

The model can be found in model_arch.py
The model:

ResnetBlock

The ResnetBlock is a building block of the VAD model. It's designed to create a residual learning path that helps in training deep networks more effectively. Here's what each part of the block does:

    1. Convolutional Layers (conv1, conv2, conv3): These layers perform 1D convolution on the input. Convolutional layers learn to extract features from the input data. The n_filters parameter determines the number of output channels (feature maps), and n_kernels is a list of kernel sizes for the three convolutional layers.
    
    2. Batch Normalization (bn1, bn2, bn3): Batch normalization normalizes the output of the convolutional layers, making the training process more stable. It helps in reducing internal covariate shift during training.
    
    3. ReLU Activation (relu1, relu2): Rectified Linear Unit (ReLU) is an activation function that introduces non-linearity to the model. It applies element-wise rectification to the output of batch normalization.
    
    4. Shortcut Connection (shortcut): This is a convolutional layer with a kernel size of 1. It's used for a shortcut connection, allowing the gradient to flow easily through the block. This helps in preventing vanishing gradients during training.
    
    5. Batch Normalization (bn_shortcut): Similar to batch normalization for the convolutional layers, this normalizes the output of the shortcut connection.
    
    6. Output Activation (out_block): The final activation function applied to the output of the block. Here, it's another ReLU activation.


Resnet1D

The Resnet1D class represents the entire VAD model and is composed of multiple ResnetBlock blocks. Here's what it does:
    
    1. Resnet Blocks (block1, block2, block3, block4): These blocks are instances of the ResnetBlock class, as explained above. They extract and process features from the input data.
    
    2. Flatten Layer (flatten): This layer is used to flatten the output of the last residual block before passing it through fully connected layers.
    
    3. Fully Connected Layers (fc1, fc2, fc3): These layers are also known as dense layers or linear layers. They take the flattened output from the previous layer and perform fully connected operations. The number of units in fc1 and fc2 can be adjusted using the n_fc_units parameter.
    
    4. ReLU Activation (relu1, relu2): ReLU activation functions introduce non-linearity to the model, applied after the fully connected layers.
    
    5. Final Output (fc3): The last fully connected layer produces the final output of the model. In the context of VAD, this output can be interpreted as a probability score or a confidence score for voice activity detection.
    
Overall, the Resnet1D model uses a residual architecture to capture complex patterns and features in the input audio data. It applies a series of convolutional and fully connected layers with activation functions to learn and represent the underlying characteristics of the audio signals. This architecture is designed to be effective for various audio processing tasks, including voice activity detection.

The ResNet architecture, which includes the ResnetBlock and Resnet1D model, has become a popular choice for various deep learning tasks, including Voice Activity Detection (VAD), due to several advantages:

    1. Residual Learning: The key innovation in ResNet is residual learning. Traditional deep networks sometimes suffer from vanishing gradients, making them difficult to train effectively when they are very deep. Residual connections in ResNet address this issue. By introducing shortcut connections (skip connections) that bypass one or more layers, gradients can flow more easily during training. This enables the training of very deep networks, which can capture complex patterns in data.
    
    2. Effective Feature Extraction: The convolutional layers in ResNet are highly effective at learning hierarchical features from the input data. In the context of audio, the 1D convolutional layers can learn important spectral and temporal features from the audio waveform. These features are essential for tasks like VAD, where discriminating between speech and non-speech regions is crucial.

    3. Batch Normalization: ResNet incorporates batch normalization layers after convolutional layers. Batch normalization normalizes the activations within each mini-batch, making training more stable and faster. It also reduces the need for careful weight initialization.
    
    4. Adaptive Learning: ReLU activation functions introduce non-linearity, allowing the model to adapt and learn complex functions. Multiple ReLU activations are applied within the ResnetBlock and Resnet1D model to introduce non-linearities at different stages of feature extraction and transformation.
    
    5. Flexibility: The ResNet architecture is flexible and can be adapted to various input sizes and types of data. For audio processing, it can be applied directly to the audio waveform or to extracted audio features like spectrograms or MFCCs.
    
    6. Strong Generalization: ResNet has demonstrated strong generalization capabilities. It can learn to extract meaningful features from data without overfitting, even when the network is very deep. This is crucial for tasks like VAD, where models need to perform well on unseen data.
    
    7. State-of-the-Art Performance: ResNet variants have achieved state-of-the-art performance in various machine learning competitions and benchmarks, including audio classification tasks.


It remains to continue the project to create the training and inference algorithm.

