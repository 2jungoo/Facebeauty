# DnCNN-based Image Denoising and Face Swapping Project

This repository provides two functionalities:
1. **Image Denoising** using the **DnCNN model** implemented in PyTorch.
2. **Face Swapping** using ONNX and InsightFace, with optional super-resolution using `realesrgan_x4plus.pth`.

Both components aim to demonstrate state-of-the-art deep learning applications for image processing.

---

## Features
### Image Denoising
- Utilizes pre-trained DnCNN models for grayscale and color image denoising.
- Supports adjustable noise levels (e.g., 15, 25, 50).
- Measures performance metrics like PSNR and SSIM.

### Face Swapping and Super-Resolution
- Detects and extracts facial features using `inswapper_128.onnx`.
- Performs face swapping between two images.
- Enhances image quality with `realesrgan_x4plus.pth` for super-resolution.

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

For Face Swapping:
Download inswapper_128.onnx from Hugging Face and place it in the project root directory.
For Super-Resolution:
Download realesrgan_x4plus.pth from Real-ESRGAN GitHub Releases and place it in the project root directory.
For DnCNN:
Download DnCNN .pth files (e.g., dncnn_25.pth) and place them in the model_zoo/ directory.

## Usage

# Image Denoising
Customize parameters via command-line arguments:

bash
코드 복사
python main.py --model_name dncnn_25 --testset_name set12 --noise_level_img 25
Additional arguments:

--show_img: Display images during processing (True or False).
Denoised images will be saved in the results/ directory.

Face Swapping
Prepare input images:

Place two images in the examples/ folder (e.g., img1.jpg, img2.jpg).
Run the face swapping script:

bash
코드 복사
python face_swapping.py
Outputs will be saved in the results/ directory.

Super-Resolution
Enhance an image using realesrgan_x4plus.pth:

bash
코드 복사
python super_resolution.py --model_path realesrgan_x4plus.pth --input_image examples/img1.jpg
Enhanced images will be saved in the results/ directory.

## Requirements
-Python >= 3.7
-numpy
-torch >= 1.1.0
-opencv-python
-matplotlib
-insightface
-onnxruntime
-realesrgan
-To install all dependencies:

bash
pip install -r requirements.txt
Models
Pre-trained Models
DnCNN Models:

Available at DnCNN GitHub.
Face Swapping Model:

Download inswapper_128.onnx from Hugging Face.
Super-Resolution Model:

Download realesrgan_x4plus.pth from Real-ESRGAN GitHub Releases.
Note:
Some model files are large (e.g., realesrgan_x4plus.pth) and cannot be included in the repository due to GitHub's file size limits. Please download them manually and place them in the appropriate directories.

## References
1. DnCNN:
   Zhang, Kai; Zuo, Wangmeng; Chen, Yunjin; Meng, Deyu; Zhang, Lei
   "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
   IEEE Transactions on Image Processing, 2017.

2.Face Swapping:
   Hugging Face - FaceSwapping Kiddo.
   
3.Super-Resolution:
 Real-ESRGAN GitHub.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

markdown
코드 복사

---

### 구성 요약
- **기능별 사용법**: Image denoising, face swapping, super-resolution 각각의 사용법을 포함.
- **필요 모델 안내**: DnCNN, `inswapper_128.onnx`, `realesrgan_x4plus.pth` 등 모델 다운로드 경로 포함.
- **요구사항**: `requirements.txt`에 기반한 라이브러리 명시.
- **디렉토리 구조 설명**: 파일 배치 및 출력 위치를 명확히 설명.
