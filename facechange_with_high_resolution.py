# 필요한 라이브러리 import
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# 애플리케이션 초기화
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))  # GPU 사용

# Face Swapper 모델 로드
model_path = 'inswapper_128.onnx'  # 모델 파일 경로
swapper = get_model(model_path, download=False, download_zip=False, providers=['CUDAExecutionProvider'])

# Real-ESRGAN 모델 초기화
sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(scale=4, model_path='RealESRGAN_x4plus.pth', model=sr_model, tile=0, tile_pad=10)

# 얼굴 교체 함수
def swap_n_show(img1_fn, img2_fn, app, swapper, upsampler, plot_before=True, plot_after=True):
    # 이미지 로드
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)

    # 교체 전 이미지 출력 (선택사항)
    if plot_before:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:, :, ::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:, :, ::-1])
        axs[1].axis('off')
        plt.show()

    # 얼굴 검출 및 특징 추출
    faces1 = app.get(img1)
    faces2 = app.get(img2)
    if len(faces1) == 0 or len(faces2) == 0:
        print("얼굴을 찾지 못했습니다.")
        return None, None

    face1 = faces1[0]
    face2 = faces2[0]

    # 얼굴 교체
    img1_swapped = swapper.get(img1, face1, face2, paste_back=True)
    img2_swapped = swapper.get(img2, face2, face1, paste_back=True)

    # 초해상도 처리 (Real-ESRGAN)
    img1_sr, _ = upsampler.enhance(img1_swapped, outscale=4)
    img2_sr, _ = upsampler.enhance(img2_swapped, outscale=4)

    # 교체 후 및 초해상도 이미지 출력 (선택사항)
    if plot_after:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1_sr[:, :, ::-1])
        axs[0].axis('off')
        axs[1].imshow(img2_sr[:, :, ::-1])
        axs[1].axis('off')
        plt.show()

    # 결과 저장
    cv2.imwrite('img1_swapped_highres.jpg', img1_sr)
    cv2.imwrite('img2_swapped_highres.jpg', img2_sr)

    return img1_sr, img2_sr

# 예제 사용
img1_path = 'min.jpg'
img2_path = 'cha.jpg'
_ = swap_n_show(img1_path, img2_path, app, swapper, upsampler)
