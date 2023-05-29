#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all necessary libraries here
from scipy.spatial import distance as dist #importing distance from scipy for computing Euclidean distance in matrix form
import numpy as np 
import imutils #importing imutils to make basic image processing functions
import cv2 #importing opencv 
import os #importing os to run the paths 
from google.colab.patches import cv2_imshow #this is to display the processed images


# In[ ]:


pip install opencv-contrib-python==4.5.3.56 #installing latest opencv packages to make sure that all the functions work


# In[ ]:


#Here we have mounted with google drive to get all the files without reuploading to Google Colab
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


MIN_CONF = 0.3 #minimum confidence of detecting object; we set this low to get more detection even in low light
NMS_THRESHOLD = 0.4 #non maximum supression threshold value we set as .4 after giving trial with different values
MIN_DISTANCE = 50 #minimum distance between two centroids in pixel value
KNOWN_DISTANCE = 1.2 #kmown distance from camera to a person in meter unit that we have used to train
PERSON_WIDTH = .41 #known width of the person 
MODEL_PATH = "/content/drive/MyDrive/YOLOv4" #used a common variable to use the following path later


# In[ ]:


#creating a function to detect people
def detect_people(frame, net, ln, personIdx=0):
	
	(H, W) = frame.shape[:2]
	results = []

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False) 
	net.setInput(blob)
	layerOutputs = net.forward(ln) 
	
	#introducing blank list for box,centroid and confidence to store the values further
	boxes = []
	centroids = []
	confidences = []
	
	for output in layerOutputs:
		
		for detection in output:
			
			scores = detection[5:] 
			classID = np.argmax(scores) #return the max value of classID of the array
			confidence = scores[classID] #the confidence value of that classID
			
			if classID == personIdx and confidence > MIN_CONF:
				
				box = detection[0:4] * np.array([W, H, W, H]) 
				(centerX, centerY, width, height) = box.astype("int")
			
				x = int(centerX - (width / 2)) 
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY)) #add centroids positions to the list
				confidences.append(float(confidence)) #store the confidences in float type
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESHOLD) #to perform non maximum supression on boxes with given threshold
	
	if len(idxs) > 0:
		
		for i in idxs.flatten():
			
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x+w,  y+h), centroids[i]) 
			results.append(r)
   	
	return results


def focal_length_finder (measured_distance, real_width, width_in_rf): #function to find focal length 
  focal_length = (width_in_rf*measured_distance)/real_width
  return focal_length

def distance_finder(focal_length, real_object_width, width_in_frame): #function to find distance from source 
  distance = (real_object_width*focal_length)/width_in_frame
  return distance


# In[ ]:


labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"]) #loading coco dataset from drive 
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([MODEL_PATH, "yolov4.weights"]) #loading yolov4 weights file from drive
configPath = os.path.sep.join([MODEL_PATH, "yolov4.cfg"])


# In[ ]:


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath) #read darknet
ln = net.getLayerNames() #getting the names of all layers of the network
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #Getting the index of the output layers


# In[ ]:


ref_person = cv2.imread('/content/ref_image.png') #read the reference image

person_data = detect_people(ref_person,net,ln)
person_width_in_rf = person_data[0][1][2]-person_data[0][1][0] #person width in reference image in pixel value
print('person width in reference image in pixel:', person_width_in_rf)

focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf) #getting focal length in pixel
print('Focal length in pixel:',focal_person)


# In[ ]:


input="/content/test_video.mp4" #select file to be processed as input
output="/content/test_video_output.avi" #setting the directory and output file name
display=1 


# In[ ]:


vs = cv2.VideoCapture(input) #define video capture path
writer = None
# loop over the frames from the video stream
count = 0 #initializing frame counts from 0
while True:

    (grabbed, frame) = vs.read() #read the selected video
    if not grabbed:
        break

    frame = imutils.resize(frame, width=700) #resizing the frames size
    results = detect_people(frame, net, ln, personIdx=LABELS.index('person')) #running detection function and storing index when detected object label is 'person'
    for d in results:
      distance = distance_finder(focal_person, PERSON_WIDTH, (d[1][2]-d[1][0])) #distance finding function
      x,y = d[2]
      cv2.putText(frame,f'd:{round(distance,2)}',(x-15,y-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    
    violate = set() # initialize the set of indexes for violating social distance

    if len(results) >= 2: #when length of detected objects are more or equal to two 
        
        centroids = np.array([r[2] for r in results]) #getting the certroids in an array of detected persons
        D = dist.cdist(centroids, centroids, metric='euclidean') #measuring the euclidean distance
        
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):                            
                if D[i, j] < MIN_DISTANCE:      #compare with the pre set min distance                                  
                    violate.add(i) #if the distance between the centroids less then min distance then add to violate set
                    violate.add(j)                    

    #loop over the results

    for (i, (prob, bbox, centroid)) in enumerate(results):       
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
       
        if i in violate:
            color = (0, 0, 255)
      
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 3, color, 2)
        
    #total number of social distancing violations on the frames

    text = 'Social Distancing Violations: {}'.format(len(violate))
    cv2.putText(frame,text,(10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX,0.90,(255, 255, 255),2,)
        
    cv2_imshow(frame) #if we want to see the frames
    cv2.imwrite(f'/content/depth_sd_frames/Frame{count}.png',frame) #if we want to write the frames to a perticular folder
    count +=1
    if display > 0:

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # if quit 'q' key pressed, break the loop
            break


    if output != '' and writer is None: # if output file name not given then have to write

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output, fourcc, 25, (frame.shape[1],frame.shape[0]), True)

    if writer is not None: # if the video writer is not None then write the frame to the output video
        writer.write(frame)


# In[ ]:


#to zip all the frames for downloading
get_ipython().system('zip -r /content/rar_frames.zip /content/rar_frames')

