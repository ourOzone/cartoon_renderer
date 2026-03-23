import numpy as np
import cv2 as cv


def quantization_color(img, k_size):
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    *_, labels, centers = cv.kmeans(data, k_size, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    color = centers[labels.flatten()].reshape(img.shape)
    return color

def get_edge(img, threshold=0.1):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dx = cv.Sobel(gray, cv.CV_64F, 1, 0) / 1020
    dy = cv.Sobel(gray, cv.CV_64F, 0, 1) / 1020
    mag = np.sqrt(dx * dx + dy * dy) / np.sqrt(2)
    edges = (mag > threshold).astype(np.uint8) * 255
    return edges


img_list = [
    "figure/f1.png",
    "figure/f2.png",
    "figure/f3.png",
    "figure/f4.png",
    "figure/f5.png",
    "figure/f6.png",
    "figure/f7.png",
    "figure/f8.png",
    "figure/f9.png"
]
img_select = 0
k_size = 10
edge_bold = 3
threshold = 0.1  # 0.0 ~ 1.0 범위

cached_img_select = -1
cached_k_size = -1
cached_color = None

while True:
    img = cv.imread(img_list[img_select])
    if img is None:
        print(f"파일 없음: {img_list[img_select]}")
        img_select = (img_select + 1) % len(img_list)
        continue

    if img_select != cached_img_select or k_size != cached_k_size:
        cached_color = quantization_color(img, k_size)
        cached_img_select = img_select
        cached_k_size = k_size

    edges = get_edge(img, threshold)

    if edge_bold == 0:
        result = cached_color.copy()
    else:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (abs(edge_bold), abs(edge_bold)))
        edges = cv.dilate(edges, kernel)
        edges_bgr = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

        if edge_bold > 0:
            result = np.where(edges_bgr > 0, 0, cached_color).astype(np.uint8)
        else:
            result = np.where(edges_bgr > 0, 255, cached_color).astype(np.uint8)

    display = result.copy()
    cv.putText(img, f"k_size: {k_size}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(img, f"threshold: {threshold:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(img, f"edge_bold: {edge_bold}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(img, f"img: {img_list[img_select]}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("cartoon", np.hstack((img, display)))
    key = cv.waitKey(0)

    if key == 27:  # ESC 종료
        break
    elif key == ord('1'):  # k_size 증가
        k_size = min(k_size + 1, 20)
        print(f"k_size: {k_size}")
    elif key == ord('2'):  # k_size 감소
        k_size = max(k_size - 1, 2)
        print(f"k_size: {k_size}")
    elif key == ord('w'):  # threshold 증가 → 엣지 적어짐
        threshold = min(round(threshold + 0.01, 2), 1.0)
        print(f"threshold: {threshold:.2f}")
    elif key == ord('s'):  # threshold 감소 → 엣지 많아짐
        threshold = max(round(threshold - 0.01, 2), 0.01)
        print(f"threshold: {threshold:.2f}")
    elif key == ord('+') or key == ord('='):  # 엣지 두께 증가
        edge_bold = min(edge_bold + 1, 20)
        print(f"edge_bold: {edge_bold}")
    elif key == ord('-'):  # 엣지 두께 감소
        edge_bold = max(edge_bold - 1, -20)
        print(f"edge_bold: {edge_bold}")
    elif key == ord('d'):  # 다음 이미지
        img_select = (img_select + 1) % len(img_list)
        print(f"img: {img_list[img_select]}")
        continue
    elif key == ord('a'):  # 이전 이미지
        img_select = (img_select - 1) % len(img_list)
        print(f"img: {img_list[img_select]}")
        continue

cv.destroyAllWindows()