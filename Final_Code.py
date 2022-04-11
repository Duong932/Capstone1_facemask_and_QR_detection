# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from pyzbar import pyzbar
import numpy as np
import datetime
import imutils
import cv2
import csv
import os

######################## Decode QR code ###############################
def read_barcodes(frame):
    barcodes = pyzbar.decode(frame)
    data_check = open("info_check.txt", mode ='r', encoding = "utf-8") 
    info_check = data_check.read()
    info_check_list = info_check.split("|")
# loop over the detected QR code locations and their corresponding locations
    for barcode in barcodes:
        x, y , w, h = barcode.rect
        #1
        barcode_info = barcode.data.decode('utf-8')
        datetime_check = datetime.datetime.now()
        # display the info and bounding box rectangle on the output frame
        cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        #2
        # the bounding box and text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, barcode_info, (x + 6, y - 6), font, 0.7, (0, 0, 255), 1)
        #3
        #processing the raw info of QR code
        data_QRinfo = barcode_info.split("|")
        data_QR = data_QRinfo[5]
        data_QR = data_QR.split("<")
        sex = str(data_QR[1])
        if (sex == str(0)):
            sex = str('Male')
        if (sex == str(1)):
            sex = str('Male')
        if (sex == str(2)):
            sex = str('Female')
        # save info QR code to txt file
        with open("barcode_result.txt", mode ='w+', encoding = "utf-8") as file:
            file.write("'")
            file.write(data_QRinfo[0])
            file.write("| ")
            file.write(data_QRinfo[1])
            file.write("| ")
            file.write(data_QRinfo[2])
            file.write("| ")
            file.write(data_QRinfo[3])
            file.write("| ")
            file.write("'")
            file.write(data_QRinfo[4])
            file.write("| ")
            file.write(sex)
            file.write("| ")
            file.write(data_QR[2])
            file.write("| ")
            file.write("'")
            file.write(data_QR[4])
            file.write("| ")
            file.write(str(datetime_check))
            file.close()
        data = open("barcode_result.txt", mode ='r+', encoding = "utf-8") 
        info = data.read()
        #print("info", info)
        data_list = info.split("|")
        # compare info of QR code 
        if (info_check_list[0] != data_list[0]):
            with open("data_QRCode.csv", mode ='a+', encoding = "utf-8") as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
                    writer.writerow(data_list)
                    file.close()
                    info_check = info 
                    print("info", info)
        #print("info_check", info_check)
        with open("info_check.txt", mode ='w+', encoding = "utf-8") as file:
            file.write(info_check)
            file.close()
    return frame

############################### Face Mask Detec #################################################
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Save infomation QR code to csv file
datetime_open = datetime.datetime.now()
print("Hello! Welcome to my Project")
data_check = open("info_check.txt", mode ='a+', encoding = "utf-8")
# Create a file csv and intial format cells 
with open("data_QRCode.csv", mode ='a+', encoding = "utf-8") as file:
    print("datime check:", datetime_open)
    datetime_open = str(datetime_open)  
    datetime_open_list = datetime_open.split()
    #print(datetime_open_list)
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(['ID number', 'Full Name', 'DoB', 'Format', 'ID QRCode PC-Covid', 
            'Sex', 'VssID' ,'Phone Number', 'Time Check', datetime_open_list[0]])
    file.close()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame_QR = vs.read()
    frame_QR = read_barcodes(frame_QR)  
    # to have a maximum width of 400 pixels  
    frame = imutils.resize(frame, width=800)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    #3
 # here it should be the pause
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
    vs.stop()
