import numpy as np
import cv2

def pos(fist):
    return (fist[0]+ fist[2]/2, fist[1] + fist[3]/2)

def dist(x,y,xx,yy):
    return np.sqrt((x - xx)**2+(y - yy)**2)

cap = cv2.VideoCapture(0)
fist_cascade = cv2.CascadeClassifier('fist.xml')
# face_cascade = cv2.CascadeClassifier('face.xml')
count = 0
timer = 50
fx = cap.get(3)/2
fy = cap.get(4)/2
rec = np.array([[fx,fy]], dtype=np.int32)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fists = fist_cascade.detectMultiScale(gray, 1.1, 5)
    if rec.shape[0] == 1:
        timer -= 1
        if timer <= 0:
            rec[0] = (fx,fy)
            timer = 50
    else:
        timer = 50
    for f in fists:
        x,y = rec[0]
        xx,yy = pos(f)
        d = dist(x,y,xx,yy)
        # print "(%d,%d) <-> (%d,%d) %lf"%(x,y,xx,yy,d)
        if d < 150 and rec.shape[0] > 0:
            rec = np.append(rec,np.asarray(pos(f))).reshape((-1,2))

    while rec.shape[0] > 30:
        rec = np.delete(rec,0,0)

    if count % 16 and rec.shape[0] > 1:
        rec = np.delete(rec,0,0)
    cv2.circle(frame,(rec[-1][0],rec[-1][1]), 15, (0,255,0),-1)

    if rec[-1][0] > fx*2-fx/2:
        print "prev"
        rec[0] = (fx,fy)

    if rec[-1][0] < fx/2:
        print "next"
        rec[0] = (fx,fy)

    for f in range(len(rec)-1):
        x,y = rec[f]
        xx,yy = rec[f+1]
        cv2.line(frame,(x,y),(xx,yy),(255,0,0),5)

    for (x,y,w,h) in fists:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(30) & 0xFF == 27:#ord('q'):
        break
    count += 1

cap.release()
cv2.destroyAllWindows()
