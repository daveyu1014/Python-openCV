import dlib
import cv2
import numpy as np
import math
import random
from PIL import ImageEnhance
predictor_path='shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
 
 
def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
 
    land_marks = []
 
    rects = detector(img_gray,0)
 
    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x,p.y] for p in predictor(img_gray,rects[i]).parts()])
        # for idx,point in enumerate(land_marks_node):
        #     # 68點座標
        #     pos = (point[0,0],point[0,1])
        #     print(idx,pos)
        #     # 利用cv2.circle給每個特徵點畫一個圈，共68個
        #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
        #     # 利用cv2.putText輸出1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)
 
    return land_marks
 
 
 
'''
利用:Interactive image warping 局部平移算法
'''
 
def localTranslationWarp(srcImg,startX,startY,endX,endY,radius):
 
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()
 
    # 計算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            #計算該點是否在形變圓的範圍之內
            #優化，第一步，直接判斷是會在（startX,startY)的矩陣框中
            if math.fabs(i-startX)>radius and math.fabs(j-startY)>radius:
                continue
 
            distance = ( i - startX ) * ( i - startX) + ( j - startY ) * ( j - startY )
 
            if(distance < ddradius):
                #計算出（i,j）座標的原座標
                #計算公式中右邊平方號裡的部分
                ratio=(  ddradius-distance ) / ( ddradius - distance + ddmc)
                ratio = ratio * ratio
 
                #映射原位置
                UX = i - ratio  * ( endX - startX )
                UY = j - ratio  * ( endY - startY )
 
                #根據雙線性插值法得到UX,UY的值
                value = BilinearInsert(srcImg,UX,UY)
                #改變當前i,j的值
                copyImg[j,i] =value
 
    return copyImg
 
 
#雙線性插值法
def BilinearInsert(src,ux,uy):
    w,h,c = src.shape
    if c == 3:
        x1=int(ux)
        x2=x1+1
        y1=int(uy)
        y2=y1+1
 
        part1=src[y1,x1].astype(np.float)*(float(x2)-ux)*(float(y2)-uy)
        part2=src[y1,x2].astype(np.float)*(ux-float(x1))*(float(y2)-uy)
        part3=src[y2,x1].astype(np.float) * (float(x2) - ux)*(uy-float(y1))
        part4 = src[y2,x2].astype(np.float) * (ux-float(x1)) * (uy - float(y1))
 
        insertValue=part1+part2+part3+part4
 
        return insertValue.astype(np.int8)
 
def face_thin_auto(src):
 
    landmarks = landmark_dec_dlib_fun(src)
 
    #如果未偵測到臉部就不動作
    if len(landmarks) == 0:
        return
 
    for landmarks_node in landmarks:
        left_landmark= landmarks_node[3]
        left_landmark_down=landmarks_node[5]
 
        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]
 
        endPt = landmarks_node[30]
 
        #計算第四點到第6點的距離作為瘦臉距離
        r_left=math.sqrt((left_landmark[0,0]-left_landmark_down[0,0])*(left_landmark[0,0]-left_landmark_down[0,0])+
                         (left_landmark[0,1] - left_landmark_down[0,1]) * (left_landmark[0,1] - left_landmark_down[0, 1]))
 
        #計算第14點到第16點的距離作為瘦臉距離
        r_right=math.sqrt((right_landmark[0,0]-right_landmark_down[0,0])*(right_landmark[0,0]-right_landmark_down[0,0])+
                         (right_landmark[0,1] -right_landmark_down[0,1]) * (right_landmark[0,1] -right_landmark_down[0, 1]))
 
        #瘦左邊臉
        thin_image = localTranslationWarp(src,left_landmark[0,0],left_landmark[0,1],endPt[0,0],endPt[0,1],r_left)
        #瘦右邊臉
        thin_image = localTranslationWarp(thin_image, right_landmark[0,0], right_landmark[0,1], endPt[0,0],endPt[0,1], r_right)
 
    #顯示
    cv2.imshow('thin',thin_image)
    cv2.imwrite('thin.jpg',thin_image)
 
#油畫效果    
def oil_effect(img):
    h, w, n = img.shape
    new_img = np.zeros((h - 2, w, n), dtype=np.uint8)
    for i in range(h - 2):
        for j in range(w):
            if random.randint(1, 10) % 3 == 0:
                new_img[i, j] = img[i - 1, j]
            elif random.randint(1, 10) % 2 == 0:
                new_img[i, j] = img[i + 1, j]
            else:
                new_img[i, j] = img[i + 2, j]
    return new_img


# 圖像增強
def img_add(img):
    enh_col = ImageEnhance.Color(img)
    color = 2.0
    new_img = enh_col.enhance(color)
    new_img = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)
    return new_img 
    
#卡通效果
def cortoon_effect(img):
    img_color = img
    for _ in range(3):
        img_color = cv2.pyrDown(img_color)
    for _ in range(7):
        img_color = cv2.bilateralFilter(img_color, 50, 50, 50)
    for _ in range(3):
        img_color = cv2.pyrUp(img_color)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=5, C=2)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    print(img_color.shape)
    print(img_edge.shape)
    new_img = cv2.bitwise_and(img_color, img_edge)
    return new_img
    

#臉部磨皮
def facial_dermabrasion_effect(img):
    
    blur_img = cv2.bilateralFilter(img, 31, 75, 75)
    #圖像融合
    result_img = cv2.addWeighted(img, 0.3, blur_img, 0.7, 0)
    cv2.imwrite("58_1.jpg", result_img)

    # 銳度
    enh_img = ImageEnhance.Sharpness(img)
    image_sharped = enh_img.enhance(1.5)
    # 對比
    con_img = ImageEnhance.Contrast(image_sharped)
    image_con = con_img.enhance(1.15)
    image_con.save("58_2.jpg")

    #img1 = cv2.imread("58.jpg")
    img2 = cv2.imread("58_2.jpg")
    #cv2.imshow("1", img1)
    cv2.imshow("2", img2)
    return img2


def main():
    img = cv2.imread('./image/me.jpg')
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
      
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imshow('src', resized)
    face_thin_auto(resized)
    cv2.waitKey(0)
 
if __name__ == '__main__':
    main()