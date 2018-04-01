#
# Copyright Amazon AWS DeepLens, 2017
#

import os
import mo
from threading import Timer
import time
import awscam
import cv2
from threading import Thread
from playsound import playsound
from botocore.session import Session
from recognitionobject import RecognitionObject
from validator import AttireValidator


ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame) 
Write_To_FIFO = True
class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)
 
    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue

def draw_bbox_and_label(frame, label, color, xmin, xmax, ymin, ymax):
      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 4)
      cv2.putText(frame, label, (xmin, ymin-15),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,128), 3)
 
valid_frame = 0;
invalid_frame = 0;
manual_check_frame = 0;

def greengrass_infinite_infer_run():
	modelPath = "deploy_ssd_resnet50_512"
        
	modelType = "ssd"
	input_width = 300
	input_height = 300
	max_threshold = 0.50
	outMap = { 1: 'Jeans', 2: 'Tee', 3: 'Blazer', 4: 'Face', 5: 'Person', 6: 'bus', 7 : 'car', 8 : 'cat', 9 : 'chair', 10 : 'cow', 11 : 'dinning table', 12 : 'dog', 13 : 'horse', 14 : 'motorbike', 15 : 'boat', 16 : 'pottedplant', 17 : 'sheep', 18 : 'sofa', 19 : 'train', 20 : 'tvmonitor' }

	#results_thread = FIFO_Thread()
	#results_thread.start()

        
        #error, model_path = mo.optimize(modelPath,input_width,input_height)

	# Load model to GPU (use {"GPU": 0} for CPU)
	mcfg = {"GPU": 1}
	model = awscam.Model("/home/aws_cam/caffe_converted_model_v4/VGG_VOC0712_SSD_300x300_deploy/VGG_VOC0712_SSD_300x300_deploy.xml", mcfg)
	#model = awscam.Model(model_path, mcfg)

	ret, frame = awscam.getLastFrame()
	if ret == False:
	    raise Exception("Failed to get frame from the stream")
	    
	yscale = float(frame.shape[0]/input_height)
	xscale = float(frame.shape[1]/input_width)

        doInfer = True
	while doInfer:
		# Get a frame from the video stream
		ret, frame = awscam.getLastFrame()
		# Raise an exception if failing to get a frame
		if ret == False:
		   raise Exception("Failed to get frame from the stream")

		# Resize frame to fit model input requirement
		frameResize = cv2.resize(frame, (input_width, input_height))

		# Run model inference on the resized frame
		inferOutput = model.doInference(frameResize)

		# Output inference result to the fifo file so it can be viewed with mplayer
		parsed_results = model.parseResult(modelType, inferOutput)['ssd']
		label = '{'
                validator = AttireValidator(frame, draw_bbox_and_label)
		for obj in parsed_results:
		    if obj['prob'] > max_threshold:
			    xmin = int( xscale * obj['xmin'] ) + int((obj['xmin'] - input_width/2) + input_width/2)
			    ymin = int( yscale * obj['ymin'] )
			    xmax = int( xscale * obj['xmax'] ) + int((obj['xmax'] - input_width/2) + input_width/2)
			    ymax = int( yscale * obj['ymax'] )
                            validator.addRecognitionObject(RecognitionObject(outMap[obj['label']], xmin, xmax, ymin, ymax))
			    label += '"{}": {:.2f},'.format(outMap[obj['label']], obj['prob'] )

                validator.processObjects()
                incrementRecognizedFrameCounter(validator)
                makeFrameDecision(frameResize)
                                
		label += '"null": 0.0'
		label += '}' 
		global jpeg
		ret,jpeg = cv2.imencode('.jpg', frame)
                fifo_path = "/tmp/results.mjpeg"
		if not os.path.exists(fifo_path):
		    os.mkfifo(fifo_path)
		f = open(fifo_path,'w')
		f.write(jpeg.tobytes())

def  writeFrameToS3AndPublishMessage(frame):
    session = Session()
    s3 = session.create_client('s3')
    file_name = 'unrecognized-images/image-'+time.strftime("%Y%m%d-%H%M%S")+'.jpg'
    ret, image = cv2.imencode('.jpg', frame)
    image_bytes = image.tobytes()
    s3Response = s3.put_object(ACL='public-read', Body=image_bytes, Bucket='deeplens-dresscode-recognition-images', Key=file_name)

    s3Url = 'https://s3.amazonaws.com/deeplens-dresscode-recognition-images/'+file_name
    message = 'Manual dress code check needed. You can view the image at ' + s3Url
    sns = session.create_client('sns', region_name='us-east-1')
    snsResponse = sns.publish(TargetArn='arn:aws:sns:us-east-1:565813021316:Manual_Dresscode_Check_Needed', 
                              Message=message
                              Subject='Manual dress check needed'
                              MessageStructure='raw')
    

def incrementRecognizedFrameCounter(validator):
    global valid_frame
    global invalid_frame
    global manual_check_frame
    if validator.isFrameValid():
       valid_frame += 1
       invalid_frame = 0
       manual_check_frame =0
    elif validator.isFrameInvalid():
       invalid_frame += 1
       valid_frame = 0
       manual_check_frame =0
    elif validator.doesFrameNeedManualCheck():
       manual_check_frame += 1
       valid_frame = 0
       invalid_frame = 0
    else:
       resetFrameCounter()

def resetFrameCounter():
    global valid_frame
    global invalid_frame
    global manual_check_frame
    valid_frame = 0
    invalid_frame = 0
    manual_check_frame =0

def makeFrameDecision(frame):
    global valid_frame
    global invalid_frame
    global manual_check_frame
    if valid_frame == 5:
      resetFrameCounter()
      playsound("/home/aws_cam/Downloads/access-granted.mp3")

    elif invalid_frame == 5:
      resetFrameCounter()
      playsound("/home/aws_cam/Downloads/access-denied.mp3")

    elif manual_check_frame == 5:
      resetFrameCounter()
      playsound("/home/aws_cam/Downloads/please-wait.mp3")
      writeFrameToS3(frame)
   
if __name__=='__main__':
	greengrass_infinite_infer_run()
        #print("mo version", mo.__version__);
