# Importing Libraries
import numpy as np
import cv2
import math

#Load Data Hand_haar_Cascade
hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

# Video Capture Part 
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret, res = cap.read() # Capture frame-by-frame 
	blur = cv2.GaussianBlur(res,(5,5),0) # Smooth the rough edges in the Image
	gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) # Convert from BGR to gray
	retval2,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # Threshold Image
	hand = hand_cascade.detectMultiScale(thresh, 1.8, 5) # Set the threshold value of the scanned hand
	mask = np.zeros(thresh.shape, np.uint8) # Creating the Mask
	for (u,z,w,h) in hand: # Marking the scanned Image
		cv2.rectangle(res,(u,z),(u+w, z+h), (170,170,170), 60) 
		cv2.rectangle(mask, (u,z), (u+w, z+h) ,255, 40)
	res2 = cv2.bitwise_and(thresh, mask)
	end = cv2.GaussianBlur(res2,(7,7),0)	
	contours, hierarchy = cv2.findContours(end, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(res, contours, -1, (255,255,0), 3)
	cv2.drawContours(end, contours, -1, (255,255,0), 3)

	if len(contours) > 0:
		cnt=contours[0]
		hull = cv2.convexHull(cnt, returnPoints=False)

		# Finding convexity faults
		fault = cv2.convexityDefects(cnt, hull)
		count_fault = 0

		# Apply the Cosine rule to find angles on all sides (Example Foromul = c2 = a2 + b2 − 2ab cos(C))
		# Ignore if angle is greater than 90 degrees
		if fault is not None:
			for i in range(fault.shape[0]):
				t,j,k,s = fault[i,0]
				finger = tuple(cnt[t][0])
				finger1 = tuple(cnt[j][0])
				tip = tuple(cnt[k][0])

				# Find the length of each edge of the triangle
				p = math.sqrt((finger1[0] - finger[0])**2 + (finger1[1] - finger[1])**2)
				z = math.sqrt((tip[0] - finger[0])**2 + (tip[1] - finger[1])**2)
				q = math.sqrt((finger1[0] - tip[0])**2 + (finger1[1] - tip[1])**2)
				c = (p+z+q)/2
				ar = math.sqrt(c*(c-p)*(c-z)*(c-q))	

				d=(2*ar)/p

				# Application Cosine rule (Example Formul = c2 = a2 + b2 − 2ab cos(C))
				angle = math.acos((z**2 + q**2 - p**2) / (2*z*q)) * 60

				# Ignore and check if the angle is greater than 90 degrees
				if angle <= 90 and d>30:
				    count_fault += 1

		# Defining the actions to be taken
		if count_fault == 1:
			cv2.putText(res,"One", (0, 50), 2, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
		elif count_fault == 2:
			cv2.putText(res, "Two", (0, 50), 2, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
		elif count_fault == 3:
			cv2.putText(res,"Three", (0, 50), 2, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
		elif count_fault == 4:
			cv2.putText(res,"Four", (0, 50), 2, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)
	cv2.imshow('res',thresh)
	cv2.imshow('res1',res)
	cv2.imshow('res2',res2)

	k = cv2.waitKey(20) & 0xff # If you are using a 64-bit machine, you should change the command line k = cv2.waitKey () to k = cv2.waitKey () &0xFF
	if k == 15:
		break
cap.release() # When everything done, release  # the video capture object
cv2.destroyAllWindows() # To close the windows we create
