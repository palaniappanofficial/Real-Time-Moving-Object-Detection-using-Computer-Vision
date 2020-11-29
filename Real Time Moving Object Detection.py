import cv2
import time
import imutils
cam=cv2.VideoCapture(3)
time.sleep(1)

firstFrame=None
area=500
while True:
    _,img=cam.read()
    text="Normal"
    img=imutils.resize(img,width=500)
    grayImg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gaussianImg=cv2.GaussianBlur(grayImg,(21,21),0)
    if firstFrame is None:
        firstFrame=gaussianImg
        continue
    imgDiff=cv2.absdiff(firstFrame,grayImg)
    threshImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    threshImg=cv2.dilate(threshImg,None,iterations=2)
    cnts=cv2.findContours(threshImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c)<500:
            continue
        (x,y,w,h)=cv2.bountingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text="Moving Object Detected"
    print(text)
    cv2.putText(img,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow("Camerafeed",img)
    key=cv2.waitkey(1) & 0xFF
    if key==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

