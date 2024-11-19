# 필요한 라이브러리 설치 및 import
import numpy as np
import cv2
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import os


# 애플리케이션 초기화
def initialize_face_swapper(model_path, det_size=(640, 640), ctx_id=0):
    """
    Initialize the face swapping application and model.

    Args:
        model_path (str): Path to the ONNX model.
        det_size (tuple): Detection size for the face analysis application.
        ctx_id (int): GPU context (set to -1 for CPU).

    Returns:
        app (FaceAnalysis): Initialized face analysis application.
        swapper (Model): Loaded face swapper model.
    """
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    swapper = get_model(model_path, download=False, download_zip=False)
    return app, swapper


# 얼굴 교체 함수
def swap_faces(img1_path, img2_path, app, swapper, plot_before=True, plot_after=True):
    """
    Perform face swapping between two images and optionally plot results.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        app (FaceAnalysis): Initialized face analysis application.
        swapper (Model): Face swapper model.
        plot_before (bool): Whether to plot images before swapping.
        plot_after (bool): Whether to plot images after swapping.

    Returns:
        img1_swapped (ndarray): Image 1 after swapping.
        img2_swapped (ndarray): Image 2 after swapping.
    """
    # 이미지 로드
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # 교체 전 이미지 출력 (선택사항)
    if plot_before:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:, :, ::-1])
        axs[0].axis('off')
        axs[0].set_title("Original Image 1")
        axs[1].imshow(img2[:, :, ::-1])
        axs[1].axis('off')
        axs[1].set_title("Original Image 2")
        plt.show()

    # 얼굴 검출 및 특징 추출
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]

    # 얼굴 교체
    img1_swapped = swapper.get(img1, face1, face2, paste_back=True)
    img2_swapped = swapper.get(img2, face2, face1, paste_back=True)

    # 교체 후 이미지 출력 (선택사항)
    if plot_after:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1_swapped[:, :, ::-1])
        axs[0].axis('off')
        axs[0].set_title("Swapped Image 1")
        axs[1].imshow(img2_swapped[:, :, ::-1])
        axs[1].axis('off')
        axs[1].set_title("Swapped Image 2")
        plt.show()

    return img1_swapped, img2_swapped


# 실행 부분
if __name__ == "__main__":
    # 모델 경로 설정
    model_path = "inswapper_128.onnx"  # ONNX 모델 경로
    app, swapper = initialize_face_swapper(model_path)

    # 사용자 입력을 통해 이미지 경로 설정
    img1_path = input("Enter the path to the first image: ")
    img2_path = input("Enter the path to the second image: ")

    # 얼굴 교체 실행
    try:
        swap_faces(img1_path, img2_path, app, swapper)
    except Exception as e:
        print(f"An error occurred: {e}")
