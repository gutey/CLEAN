# Channel- and Frequency-Invariant EEG Artifact Removal Transformer (CLEAN)

This repository contains the code for the **Channel- and Frequency-Invariant EEG Artifact Removal Transformer (CLEAN)**, a deep learning-based method for removing artifacts from EEG signals. The approach is designed to be flexible with respect to input channel configurations and sampling frequencies, making it adaptable to a wide range of EEG datasets and downstream tasks.

> **Note:** In some parts of the code or documentation, you might encounter the term **UPT4EEG**. This was the previous name for **CLEAN** and can be considered equivalent.


## 📖 Abstract
Electroencephalography (EEG) is a widely used method for recording brain activity, but it is highly susceptible to artifacts. This makes artifact removal essential for reliable downstream tasks such as epileptic seizure detection or Brain-Computer Interface (BCI). However, existing artifact removal methods are often inflexible with respect to input channel configurations and sampling frequencies, limiting their adaptability and range of application. 

Therefore, this thesis introduces **CLEAN** – a **ChanneL-** and **frequency-independent Eeg Artifact removal traNsformer**. The proposed approach reconstructs EEG data for any channel configuration and at any sampling frequency. This adaptability is achieved by sampling data points at random channels and time steps during training, combined with positional and temporal encoding. Moreover, the encoder of the proposed model learns a field representation of EEG data in the latent space, enabling a perceiver block in the decoder to query denoised signals at arbitrary positions and frequencies. This approach ensures high flexibility across diverse datasets and downstream tasks. 

Experimental results demonstrate that the model outperforms existing deep learning-based artifact removal methods while offering the added benefits of channel- and frequency-independence.

---

## 🛠️ Setup
To set up the project and install the required dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/gutey/CLEAN.git
   cd CLEAN
   ```

2. Create and activate environment:
   ```bash
   conda create --name clean python=3.9
   conda activate clean
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Run the Model

To run the model, follow these steps:

1. **Train the model**:
   To train the model, use the following command:
   ```bash
   python main.py --config config.yaml
   ```
2. **Evaluate the model**:
   After training, you can evaluate the model using the following command:
   ```bash
   python evaluation/inference.py --model checkpoint.pth --test_data test_data/
   ```
   


## 📄 License

This project is licensed under the [MIT License](LICENSE).
