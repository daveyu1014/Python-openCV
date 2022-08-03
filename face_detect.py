import cv2


#人臉馬賽克效果
def mosaic(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 影像轉換成灰階
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # 載入人臉偵測模型
    faces = face_cascade.detectMultiScale(gray,1.2,3)  # 開始辨識影像中的人臉
    
    for (x, y, w, h) in faces:
        mosaic = img[y:y+h, x:x+w]   # 馬賽克區域
        level = 15                   # 馬賽克程度
        mh = int(h/level)            # 根據馬賽克程度縮小的高度
        mw = int(w/level)            # 根據馬賽克程度縮小的寬度
        mosaic = cv2.resize(mosaic, (mw,mh), interpolation=cv2.INTER_LINEAR) # 先縮小
        mosaic = cv2.resize(mosaic, (w,h), interpolation=cv2.INTER_NEAREST)  # 然後放大
        img[y:y+h, x:x+w] = mosaic   # 將指定區域換成馬賽克區域
    
    return img

def face_detect(img):
    
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")          # 使用眼睛模型
    mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")  # 使用嘴巴模型
    nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")    # 使用鼻子模型
    
    gray = cv2.medianBlur(img, 1)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    eyes = eye_cascade.detectMultiScale(gray)      # 偵測眼睛
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mouths = mouth_cascade.detectMultiScale(gray)  # 偵測嘴巴
    for (x, y, w, h) in mouths:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    noses = nose_cascade.detectMultiScale(gray)    # 偵測鼻子
    for (x, y, w, h) in noses:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return img

def main():
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(frame,(540,320))
        
    
        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break     # 按下 q 鍵停止
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()