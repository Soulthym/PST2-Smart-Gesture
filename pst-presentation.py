import numpy as np
import cv2
from vk import *

def right():
    press('right_arrow')

def left():
    press('left_arrow')

def zoom():
    pressHoldRelease('ctrl','+')

def dezoom():
    pressHoldRelease('ctrl','-')

def pos(fist):
    return (fist[0]+ fist[2]/2, fist[1] + fist[3]/2)

def dist(P1,P2):
    return np.sqrt((P1[0] - P2[0])**2+(P1[1] - P2[1])**2)

def newBall(fx,fy):
    return np.array([fx,fy], dtype=np.int32)

def stepBall(frame,fists,B,timer,fx,fy):
    if timer < 0:
        timer = 50
        B = newBall(fx,fy)
    else :
        timer -= 1
    for f in fists:
        F = pos(f)
        if dist(B,F) < 75:
            timer = 50
            B = F
    if len(B)>0 :
        cv2.circle(frame,(B[0],B[1]), 15, (0,255,0),-1)
    return B, timer

def isin(P,R):
    cv2.rectangle(frame,(R[0],R[2]),(R[1],R[3]),R[5],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,R[4],(R[0]+15,R[2]+25), font, 0.7, R[5],2)
    cv2.putText(frame,R[6],(R[0]+15,R[2]+50), font, 0.7, R[5],2)
    if P[0] >= R[0] and P[0] <= R[1] and P[1] >= R[2] and P[1] <= R[3]:
        print "in",R[4]
        return 1

timer = 50
cap = cv2.VideoCapture(0)
fist_cascade = cv2.CascadeClassifier('fist.xml')
fx = cap.get(3)/2
fy = cap.get(4)/2
x1, x2, x3, x4, x5, x6, x7, x8, x9 = [(int)(i*fx/4) for i in range(9)]
y1, y2, y3, y4, y5, y6, y7, y8, y9 = [(int)(i*fy/4) for i in range(9)]
Left =      (x1,x2,y2,y8,"left",(180,105,255),"->")
Right =     (x8,x9,y2,y8, "right",(212,255,127),"<-")
Top =       (x4,x6,y1,y3,"top",(211,0,148),"zoom in")
Bottom =    (x4,x6,y7,y9,"bottom",(0,140,255),"zoom out")

B = newBall(fx,fy)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fists = fist_cascade.detectMultiScale(gray, 1.1, 6)

    B,timer = stepBall(frame,fists,B,timer,fx,fy)
    if isin(B,Left):
        right()
        # left()
        B = newBall(fx,fy)
    if isin(B,Right):
        left()
        # right()
        B = newBall(fx,fy)
    if isin(B,Top):
        zoom()
        B = newBall(fx,fy)
    if isin(B,Bottom):
        dezoom()
        B = newBall(fx,fy)

    for (x,y,w,h) in fists:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(30) & 0xFF == 27:#ord('q'):#27:
        break

cap.release()
cv2.destroyAllWindows()
