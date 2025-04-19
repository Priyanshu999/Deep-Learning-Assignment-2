## 🚀 Usage

### 🔹 Running Training Script

To train the neural network with customizable hyperparameters, execute the following command:

```bash
python train.py --num_filters 64 --epochs 10 --activation ReLU

| Name                                | Default Value | Description |
|-------------------------------------|---------------|-------------|
| `-nf`, `--num_filters`              | `32`          | Number of filters in the first convolutional layer. |
| `-fo`, `--filter_organisation`      | `double`      | How filter count changes across layers. <br>Choices: `same`, `double`, `half` |
| `-dr`, `--dropout_rate`             | `0`           | Dropout rate used in training. |
| `-e`, `--epochs`                    | `5`           | Number of epochs to train the neural network. |
| `-fs`, `--filter_size`              | `5`           | Size of the convolutional filters. |
| `-neu`, `--dense_neurons`           | `512`         | Number of neurons in the dense (fully connected) layer. |
| `-a`, `--activation`                | `GELU`        | Activation function to use. <br>Choices: `ReLU`, `GELU`, `LeakyReLU`, `SiLU`, `Mish` |
| `-da`, `--data_augmentation`        | `True`        | Whether to use data augmentation. |
| `-bn`, `--batch_norm`               | `True`        | Whether to use batch normalization. |
