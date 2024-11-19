# Facechange wiht high resolution

This repository provides two functionalities:
1. **Image Denoising** using the **DnCNN model** implemented in PyTorch.
2. **Face Swapping** using ONNX and InsightFace, with super-resolution using `realesrgan_x4plus.pth`.

Both components aim to demonstrate state-of-the-art deep learning applications for image processing.

---

## Features

### Face Swapping and Super-Resolution
- Detects and extracts facial features using `inswapper_128.onnx`.
- Performs face swapping between two images.
- Enhances image quality with `realesrgan_x4plus.pth` for super-resolution.

### Image Denoising
- Utilizes pre-trained DnCNN models for grayscale and color image denoising.
- Supports adjustable noise levels (e.g., 15, 25, 50).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-processing-project.git
   cd image-processing-project

2.Install the required Python packages:

   ```bash
      pip install -r requirements.txt
   ```

3.Download the necessary models:

**For Face Swapping:**

Download inswapper_128.onnx from Hugging Face and place it in the project root directory.

**For Super-Resolution:**

Download realesrgan_x4plus.pth from Real-ESRGAN GitHub Releases and place it in the project root directory.

**For DnCNN:**

Download DnCNN .pth files (e.g., dncnn_25.pth) and place them in the model_zoo/ directory.
You have to download [DnCNN GitHub Repository.](https://github.com/cszn/DnCNN)
## Usage

## Face Swapping & high resolution
1.Prepare input images:

Place two images in the examples/ folder (e.g., img1.jpg, img2.jpg).

2.Run the face swapping script:

```bash
python facechange_with_high_resolution.py
```

Super-Resolution
Enhance an image using realesrgan_x4plus.pth

Outputs will be saved in the results/ directory.

## DnCNN (Image Denoising)
1.Prepare your environment: 

   Ensure you have the main.py script in your project directory.  Additio1.nal utility files can be downloaded from the official DnCNN repository.

2.Download required files:

   Download the required utility files (utils_logger.py, utils_model.py, utils_image.py) and pre-trained models (dncnn_25.pth, etc.) from the [DnCNN GitHub Repository.](https://github.com/cszn/DnCNN)
   Place the utility files in the utils/ folder and the pre-trained model files in the model_zoo/ folder.

3.Run the DnCNN script: Execute the denoising script using the following command:
  ``` bash
   python main.py --model_name dncnn_25 --testset_name set12 --noise_level_img 25
```

## Requirements
```bash
pip install -r requirements.txt
```

optional
```bash
pip install -r requirements_DnCNN.txt
```

## Models
## Pre-trained Models
1.DnCNN Models:
   [Available at DnCNN GitHub.](https://github.com/cszn/DnCNN)

2.Face Swapping Model:
   [Download inswapper_128.onnx from Hugging Face.](https://huggingface.co/kiddobellamy/faceswapping_kiddo)

3.Super-Resolution Model:
   [Download realesrgan_x4plus.pth from Real-ESRGAN GitHub Releases.](https://github.com/xinntao/Real-ESRGAN)

## Note:
Some model files are large (e.g., realesrgan_x4plus.pth) and cannot be included in the repository due to GitHub's file size limits. Please download them manually and place them in the appropriate directories.

## References
1. DnCNN:
   Zhang, Kai; Zuo, Wangmeng; Chen, Yunjin; Meng, Deyu; Zhang, Lei
   "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
   IEEE Transactions on Image Processing, 2017.

2.Face Swapping:
   [Hugging Face - FaceSwapping Kiddo.](https://huggingface.co/kiddobellamy/faceswapping_kiddo)
   
3.Super-Resolution:
[ Real-ESRGAN GitHub.](https://github.com/xinntao/Real-ESRGAN)

