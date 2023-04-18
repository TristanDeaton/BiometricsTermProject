
import cv2
import numpy as np

def findHands(frame, averageBlueSkinColorValue, averageGreenSkinColorValue, averageRedSkinColorValue, scaleFactor, pastHands, handNum, offsetx, offsety):
    avgH1X = int((pastHands[handNum][0][0]+ pastHands[handNum][1][0] + pastHands[handNum][2][0]) / 3) - int(offsetx)
    avgH1Y = int((pastHands[handNum][0][1]+ pastHands[handNum][1][1] + pastHands[handNum][2][1]) / 3) - int(offsety)
    skinLocationY = []
    skinLocationX = []
    for xValueIterator in range(0, len(frame[0]), 1):
        for yValueIterator in range(0, len(frame), 1):
            blue = frame[yValueIterator][xValueIterator][0]
            green = frame[yValueIterator][xValueIterator][1]
            red = frame[yValueIterator][xValueIterator][2]
            if(((blue > averageBlueSkinColorValue * (1 - scaleFactor)) and (blue < averageBlueSkinColorValue * (1 + scaleFactor))) and ((green > averageGreenSkinColorValue * (1 - scaleFactor)) and (green < averageGreenSkinColorValue * (1 + scaleFactor))) and ((red > averageRedSkinColorValue * (1 - scaleFactor)) and (red < averageRedSkinColorValue * (1 + scaleFactor)))):
                skinLocationY.append(yValueIterator)
                skinLocationX.append(xValueIterator)
    centroidX = avgH1X
    centroidY = avgH1Y
    sumX = avgH1X
    sumY = avgH1Y
    sumCounter = 1
    lengthOfSkin = len(skinLocationY)
    for i in range(0, (lengthOfSkin),1):
        sumX = sumX + skinLocationX[i]
        sumY = sumY + skinLocationY[i]
        sumCounter = sumCounter + 1
    centroidX = sumX / sumCounter
    centroidY = sumY / sumCounter
    return int(centroidX), int(centroidY)

def findShoulder(frame, headXmin, headXmax):
    edge = cv2.Canny(frame, 130,200)
    shoulderValues = np.zeros(len(edge[0]))
    for x in range(0, len(edge[0]), 1):
        edgeCheck = False
        headCheck = False
        if((x > headXmin - 5) and (x < headXmax + 5)):
            headCheck = True
        for y in range(0, len(edge), 1):
            if(edge[y][x] == 255):
                edgeCheck = True
            if(edgeCheck and not(headCheck)):
                shoulderValues[x] = shoulderValues[x] + 1
    # print(shoulderValues)
    cv2.imshow("shoulders", edge)
    cv2.waitKey(1)
    leftCounter = 0
    rightCounter = 0
    maxValue = len(edge[0])
    maxY = len(edge)
    shoulderDifference = 20
    for x in range(0, int(maxValue / 2), 1):
        if(shoulderValues[x] > 0):
            leftCounter = leftCounter + 1
        if(shoulderValues[maxValue-x - 1] > 0):
            rightCounter = rightCounter + 1
        if(rightCounter > (shoulderDifference + 1)  and leftCounter > (shoulderDifference + 1)):
            break
        if(rightCounter == shoulderDifference):
            rightShoulderX = maxValue - x - 1
            rightShoulderY = maxY - shoulderValues[maxValue-x]
        if(leftCounter == shoulderDifference):
            leftShoulderX = x
            leftShoulderY = maxY - shoulderValues[maxValue-x]
    return int(leftShoulderX), int(leftShoulderY), int(rightShoulderX), int(rightShoulderY)


def detectShrug(pastShoulders, shoulderXL, shoulderYL, shoulderXR, shoulderYR):
    scaleFactor = 0.12
    averageShoulderXL = int((pastShoulders[0][0] + pastShoulders[1][0] + pastShoulders[2][0] + pastShoulders[3][0] + pastShoulders[4][0])/ 5)
    averageShoulderYL = int((pastShoulders[0][1] + pastShoulders[1][1] + pastShoulders[2][1] + pastShoulders[3][1] + pastShoulders[4][1])/ 5)
    averageShoulderXR = int((pastShoulders[0][2] + pastShoulders[1][2] + pastShoulders[2][2] + pastShoulders[3][2] + pastShoulders[4][2])/ 5)
    averageShoulderYR = int((pastShoulders[0][3] + pastShoulders[1][3] + pastShoulders[2][3] + pastShoulders[3][3] + pastShoulders[4][3])/ 5)
    if(((shoulderXL > (averageShoulderXL * (1 - scaleFactor))) and (shoulderXL < (averageShoulderXL * (1 + scaleFactor)))) and 
       (((shoulderYL > (averageShoulderYL * (1 - scaleFactor))) and (shoulderYL < (averageShoulderYL * (1 + scaleFactor))))) and 
       (((shoulderXR > (averageShoulderXR * (1 - scaleFactor))) and (shoulderXR < (averageShoulderXR * (1 + scaleFactor))))) and 
       (((shoulderYR > (averageShoulderYR * (1 - scaleFactor))) and (shoulderYR < (averageShoulderYR * (1 + scaleFactor)))))):
        return False
    else:
        print("shrug")
        return True

def main():
    scaleFactor = 0.22
    faceChecker = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    videoToUse = 'Video3.mp4'
    outputVideo = 'output - ' + videoToUse
    video = cv2.VideoCapture(videoToUse)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    settings = cv2.VideoWriter_fourcc(*'MP4V')
    ouput = cv2.VideoWriter(outputVideo, settings, fps, (int(width), int(height)), True)
    frameNumber = 0
    averageRedSkinColorValue = -1
    averageGreenSkinColorValue = -1
    averageBlueSkinColorValue = -1
    pastHeadPositions = [[564,71],[564,70],[563,70]]
    pastHandPositions = [[603,603],[603,603],[603,603]],[[750,600],[750,600],[750,600]]
    baseShoulder = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    scaleFactorFace = 0.15
    lengthShoulder = 300
    shrug = False
    while video.isOpened():
        print("Frame currently: " + str(frameNumber) + " and seconds are: " + str(int(frameNumber / fps)))
        ret, frame = video.read()
        if(ret == True):
            imgInGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            avgXFace = int((pastHeadPositions[0][0]+ pastHeadPositions[1][0] + pastHeadPositions[2][0]) / 3)
            avgYFace = int((pastHeadPositions[0][1]+ pastHeadPositions[1][1] + pastHeadPositions[2][1]) / 3)
            minXFace = int((avgXFace + 30) * (1 - scaleFactorFace))
            maxXFace = int((avgXFace + 100) * (1 + scaleFactorFace))
            minYFace = int((avgYFace - 30) * (1 - scaleFactorFace))
            maxYFace = int((avgYFace + 140) * (1 + scaleFactorFace))
            limitedImgForFace = imgInGray[minYFace:maxYFace,minXFace:maxXFace]
            faces = faceChecker.detectMultiScale(limitedImgForFace, 1.15, 15)
            shoulderMinX = pastHeadPositions[2][0]
            shoulderMinY = pastHeadPositions[2][1]
            shoulderMaxX = pastHeadPositions[2][0]
            shoulderMaxY = pastHeadPositions[2][1]
            headXmin = 0
            headXmax = 0
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, (x+minXFace,y+minYFace), (x+minXFace+w, y+minYFace+h), (255, 0, 0), 2)
                face_color = np.mean(frame[y+minYFace+10:y+minYFace+h-10, x+minXFace+10:x+minXFace+w-10], axis = (0, 1))
                redValue = int(face_color[2])
                greenValue = int(face_color[1])
                blueValue = int(face_color[0])
                pastHeadPositions.append(pastHeadPositions.pop(0))
                pastHeadPositions[2][0] = x+minXFace
                pastHeadPositions[2][1] = y+minYFace
                shoulderMinY = y + minYFace + h - 40
                shoulderMinX = x + minXFace - 120
                shoulderMaxY = y + minYFace + h + 70
                shoulderMaxX = x + minXFace + w + 120
                headXmin = x+minXFace
                headXmax = x+minXFace+w
            if(averageBlueSkinColorValue == -1 and averageBlueSkinColorValue == -1 and averageGreenSkinColorValue == -1):
                averageRedSkinColorValue = redValue
                averageGreenSkinColorValue = greenValue
                averageBlueSkinColorValue = blueValue
            elif(frameNumber < 20):
                averageRedSkinColorValue = int(((averageRedSkinColorValue * frameNumber) + redValue) / (frameNumber + 1))
                averageGreenSkinColorValue = int(((averageGreenSkinColorValue * frameNumber) + greenValue) / (frameNumber + 1))
                averageBlueSkinColorValue = int(((averageBlueSkinColorValue * frameNumber) + blueValue) / (frameNumber + 1))
            # for x in range(shoulderMinX, shoulderMaxX,1):
            #     for y in range(shoulderMinY, shoulderMaxY,1):
            #         frame[y][x][0] = 0
            #         frame[y][x][1] = 0
            #         frame[y][x][2] = 255
            shoulderFrame = frame[shoulderMinY:shoulderMaxY,shoulderMinX:shoulderMaxX]
            shoulderXL, shoulderYL, shoulderXR, shoulderYR = findShoulder(shoulderFrame, headXmin - shoulderMinX, headXmax - shoulderMinX)
            shoulderXL = shoulderXL + shoulderMinX
            shoulderYL = shoulderYL + shoulderMinY
            shoulderXR = shoulderXR + shoulderMinX
            shoulderYR = shoulderYR + shoulderMinY
            # print(shoulderXL)
            # print(shoulderYL)
            # print(shoulderXR)
            # print(shoulderYR)
            if(frameNumber < 5):
                baseShoulder[frameNumber] = [shoulderXL, shoulderYL, shoulderXR, shoulderYR]
            else:
                shrug = detectShrug(baseShoulder, shoulderXL, shoulderYL, shoulderXR, shoulderYR)
            if(shrug):
                colorShoulder = (0,255,0)
            else:
                colorShoulder = (0,0,255) 
            cv2.line(frame,(shoulderXL,shoulderYL),(shoulderXR,shoulderYR),colorShoulder, 3)


            handNum = 0
            handW = 40
            handH = 40
            scaleFactorHands = 0.15
            averageHandLeftX = int((pastHandPositions[handNum][0][0]+ pastHandPositions[handNum][1][0] + pastHandPositions[handNum][2][0]) / 3)
            averageHandLeftY = int((pastHandPositions[handNum][0][1]+ pastHandPositions[handNum][1][1] + pastHandPositions[handNum][2][1]) / 3)
            startingHeightLeft = int((averageHandLeftY - 20) * (1 - scaleFactorHands))
            startingWidthLeft = int((averageHandLeftX - 20) * (1 - scaleFactorHands))
            goToHeightLeft = int((averageHandLeftY + 50) * (1 + scaleFactorHands))
            goToWidthLeft = int((averageHandLeftX + 50) * (1 + scaleFactorHands))
            goToWidthLeft = goToWidthLeft - int((goToWidthLeft - startingWidthLeft) / 2) + 40
            if(goToHeightLeft > height):
                goToHeightLeft = int(height)
            smallerFrameLeft = frame[startingHeightLeft:goToHeightLeft, startingWidthLeft:goToWidthLeft]
            positionX, positionY = findHands(smallerFrameLeft, averageBlueSkinColorValue, averageGreenSkinColorValue, averageRedSkinColorValue, scaleFactor, pastHandPositions, handNum, startingHeightLeft, startingWidthLeft)
            pastHandPositions[handNum].append(pastHandPositions[handNum].pop(0))
            pastHandPositions[handNum][2][0] = positionX + startingHeightLeft
            pastHandPositions[handNum][2][1] = positionY + startingWidthLeft
            cv2.rectangle(frame, (positionX - handW + startingWidthLeft, positionY - handH + startingHeightLeft), (positionX+handW + startingWidthLeft, positionY + startingHeightLeft + handH), (255, 255, 255), 2)
            handNum = 1
            averageHandRightX = int((pastHandPositions[handNum][0][0]+ pastHandPositions[handNum][1][0] + pastHandPositions[handNum][2][0]) / 3)
            averageHandRightY = int((pastHandPositions[handNum][0][1]+ pastHandPositions[handNum][1][1] + pastHandPositions[handNum][2][1]) / 3)
            startingHeightRight = int((averageHandRightY - 20) * (1 - scaleFactorHands))
            startingWidthRight = int((averageHandRightX + 30) * (1 - scaleFactorHands))
            goToHeightRight = int((averageHandRightY + 50) * (1 + scaleFactorHands))
            goToWidthRight = int((averageHandRightX + 100) * (1 + scaleFactorHands))
            if(goToHeightRight > height):
                goToHeightRight = int(height)
            smallerFrameLeft = frame[startingHeightRight:goToHeightRight, startingWidthRight:goToWidthRight]
            positionX, positionY = findHands(smallerFrameLeft, averageBlueSkinColorValue, averageGreenSkinColorValue, averageRedSkinColorValue, scaleFactor, pastHandPositions, handNum, startingHeightRight, startingWidthRight)
            pastHandPositions[handNum].append(pastHandPositions[handNum].pop(0))
            pastHandPositions[handNum][2][0] = positionX + startingHeightRight
            pastHandPositions[handNum][2][1] = positionY + startingWidthRight
            cv2.rectangle(frame, (positionX - handW + startingWidthRight, positionY - handH + startingHeightRight), (positionX+handW + startingWidthRight, positionY + startingHeightRight + handH), (0, 255, 0), 2)

            ouput.write(frame)
            frameNumber = frameNumber + 1
        else:
            video.release()
            ouput.release()
main()