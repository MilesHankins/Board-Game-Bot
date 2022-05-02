import numpy as np
import time
import cv2

GAMEID={0:"CHECKERS", 1:"CHINESE CHECKERS", 2:"SOLITAIRE", 3:"CONNECT FOUR"}

CCBOARDMIN=np.array([40,0,100])
CCBOARDMAX=np.array([120,120,255])


CBORDERMIN=np.array([0,128,0])
CBORDERMAX=np.array([30,255,255])
CSQUARERMIN=np.array([100,0,100])
CSQUARERMAX=np.array([162,255,255])
CSQUAREBMIN=np.array([0,0,0])
CSQUAREBMAX=np.array([180,255,60])

C4BORDERYMIN=np.array([85,128,100])
C4BORDERYMAX=np.array([120,255,255])


def IdentifyGame(inimg):
    gray = cv2.cvtColor(inimg, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(inimg, cv2.COLOR_RGB2HSV)
    Scores=[0,0,0,0]
    
    #PART 1 Checkers
    Scores[0]=CheckersCHK(inimg,gray,hsv)
    
    #PART 2 Chinese Checkers
    Scores[1]=ChineseCheckersCHK(inimg,gray,hsv)
    
    #PART 3 Solitaire
    Scores[2]=SolitaireCHK(inimg,gray,hsv)
    
    #PART 4 Connect Four
    Scores[3]=ConnectFourCHK(inimg,gray,hsv)

    print(Scores)
    return Scores

def CheckersCHK(inimg,gray,hsv):
    mask = cv2.inRange(hsv, CBORDERMIN, CBORDERMAX)
    maskR = cv2.inRange(hsv, CSQUARERMIN, CSQUARERMAX)
    maskB = cv2.inRange(hsv, CSQUAREBMIN, CSQUAREBMAX)
    maskRB=maskR+maskB
    cv2.medianBlur(maskRB,11)
    _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    _, contoursRB, _ = cv2.findContours(maskRB, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    contour_list = []
    maxArea=0.01
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if(w>450 and h>450 and w*h>maxArea):
            maxArea=w*h
    maxScore=0
    drawList=[]
    for contour in contoursRB:
        x,y,w,h = cv2.boundingRect(contour)
        if(w>450 and h>450):
            drawList.append(contour)
            if(w*h<maxArea):
                maxScore=max(maxScore,w*h/maxArea)
    return maxScore

def ChineseCheckersCHK(inimg,gray,hsv):
    mask = cv2.inRange(hsv, CCBOARDMIN, CCBOARDMAX)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask,kernel, iterations=2)
    mask = cv2.GaussianBlur(mask,(11,11),0)
    _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        if ((len(approx) > 8) and (area > 30) and (w>250)):
            contour_list.append(contour)
    maxVal=0.0
    for cnt in contour_list:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        maxA=3.14*radius*radius
        maxVal=max(maxVal,(area/maxA))
    if maxVal<.65:
        return 0
    return maxVal


def SolitaireCHK(inimg,gray,hsv):
    gray=cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    mask = cv2.inRange(inimg, np.array([128,128,128]), np.array([255,255,255]))
    _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    contour_list=[]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w>50 and h>50:
            contour_list.append(contour)
    Range=0.30
    scoreList=[0]
    for i in range(len(contour_list)):
        best_ratio=0
        rect1 = cv2.minAreaRect(contour_list[i])
        w1=rect1[1][0]
        h1=rect1[1][1]
        count=0
        for j in range(len(contour_list)):
            if i != j:
                rect2 = cv2.minAreaRect(contour_list[j])
                w2=rect2[1][0]
                h2=rect2[1][1]
                if(w2>w1*(1.0-Range) and w2<w1*(1.0+Range) and h2>h1*(1.0-Range) and h2<h1*(1.0+Range)):
                    count+=1
                    best_ratio=max(best_ratio,min((w1*h1)/(w2*h2),(w2*h2)/(w1*h1)))
        if count>2:
            scoreList.append(best_ratio)
    return max(scoreList)
   
def ConnectFourCHK(inimg,gray,hsv):
    mask = cv2.inRange(hsv, C4BORDERYMIN, C4BORDERYMAX)
    _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    maxScore=0
    contour_list=[]
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if w>200 and h>200:
            contour_list.append(contour)
            maxScore=max(maxScore,min((w/h)/1.2,1.2/(w/h)))
    fg = cv2.bitwise_or(inimg, inimg, mask=mask)
    cv2.drawContours(fg, contour_list,  -1, (255,0,0), 2)
    cv2.imshow("Temp Window",fg)
    return maxScore    
    
    
