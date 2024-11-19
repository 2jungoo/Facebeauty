# Face Swapping and Super-Resolution Project

This repository provides an implementation of face swapping using ONNX and InsightFace, along with image super-resolution using the `realesrgan_x4plus.pth` model.
If the user wishes, DnCNN can be utilized additionally.
---

## Features
- **Facial Detection**: Detects and extracts facial features from images.
- **Face Swapping**: Performs face swapping using the `inswapper_128.onnx` model.
- **Super-Resolution**: Enhances image quality with `realesrgan_x4plus.pth`.

---

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-swapping-and-super-resolution.git
   cd face-swapping-and-super-resolution

2.Install the required Python packages:
   ```bash
   pip install -r requirements.txt

3.Download the necessary models:

Face Swapping Model: Download inswapper_128.onnx from Hugging Face and place it in the project root directory.
Super-Resolution Model: Download realesrgan_x4plus.pth from Real-ESRGAN GitHub Releases and place it in the project root directory.
---
## Usage
1.Prepare your input images:
Place two images in the examples/ folder (e.g., img1.jpg, img2.jpg).

2.Run the face swapping script:

```bash
   python face_swapping.py

3.For super-resolution:
   '''bash
      python super_resolution.py --model_path realesrgan_x4plus.pth --input_image examples/img1.jpg
4.Outputs will be saved in the results/ directory.

---
##Model Files
inswapper_128.onnx: Required for face swapping. Download from Hugging Face.
realesrgan_x4plus.pth: Required for super-resolution. Download from Real-ESRGAN GitHub Releases.
