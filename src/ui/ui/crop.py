import cv2 as cv
import numpy as np


def detect_table_and_crop(image):
    
    # 1. grayscale 변환
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 2. blur (노이즈 감소)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    # 3. 밝은 영역 threshold (흰색 책상)
    _, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    # 4. morphology (작은 노이즈 제거)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 31))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel, iterations=3)


    # 5. contour 찾기
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("책상 영역을 찾지 못했습니다.")
        return None, None

    # 6. 가장 큰 contour = 책상
    table_contour = max(contours, key=cv.contourArea)

    # 7. bounding box 생성
    x, y, w, h = cv.boundingRect(table_contour)

    # 8. crop
    cropped = image[y:y+h, x:x+w]

    return cropped, (x, y, w, h)


import os

if __name__ == "__main__":

    path = "./data/"
    outpath = "./output/"
    for i in range(1,7):
        filename=f"img{i:03d}.jpg"
        image = cv.imread(os.path.join(path, filename))
        
        # cropped, bbox = detect_table_and_crop(image)

        # if cropped is None:
        #     exit()

        # x, y, w, h = bbox

        # 시각화
        # vis = image.copy()
        # cv.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 20)

        # cv.imwrite(os.path.join(outpath, filename),cropped)