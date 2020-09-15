import cv2
import imutils
import pytesseract

image=cv2.imread('1234.jpg')

image = imutils.resize(image, width = 500 )

cv2.imshow("original",image)
cv2.waitKey(0)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale",gray)
cv2.waitKey(0)
gray=cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("bilateralFilter",gray)
cv2.waitKey(0)
edged=cv2.Canny(gray,170,200)
cv2.imshow("Canny_edges",edged)
cv2.waitKey(0)

blur = cv2.GaussianBlur(edged,(5,5),0)
cv2.imshow("smooth",blur)
cv2.waitKey(0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
#retval,thresh = cv2.threshold(gray,48,255,cv2.THRESH_BINARY)
cv2.imshow("adaptiveThreshold",thresh)
cv2.waitKey(0)


_, cnts, new = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

img1=image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)
cv2.imshow("ALL Contours",img1)
cv2.waitKey(0)

cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:30]
NumberPlateCnt = None


img2=image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("Top 30 Contours",img2)
cv2.waitKey(0)


count=0
idx=7
for c in cnts:
	peri=cv2.arcLength(c,True)
	approx=cv2.approxPolyDP(c,0.02*peri,True)

	if len(approx) == 4 :
 		NumberPlateCnt = approx
 		x, y, w, h = cv2.boundingRect(c)
 		new_img = thresh[y:y + h, x:x + w]
 		cv2.imwrite('Cropped Images-Text/'+str(idx)+'.png',new_img)
 		idx+=1
 		break
cv2.drawContours(image,[NumberPlateCnt],-1,(0,255,0),3)
cv2.imshow("final img - number plated detected",image)
cv2.waitKey(0)

Cropped_img_loc = 'Cropped Images-Text/7.png'
cv2.imshow("Cropped Image",cv2.imread(Cropped_img_loc))
cv2.waitKey(0)

text=pytesseract.image_to_string(Cropped_img_loc,lang='eng')
print("number is :",text)
cv2.waitKey(0)







#read image
img = cv2.imread(Cropped_img_loc)

#grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

#binarize 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.waitKey(0)

#find contours
im2,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y+h, x:x+w]

    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)

cv2.imshow('marked areas',img)
cv2.waitKey(0)