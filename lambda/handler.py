#
# Copyright Amazon AWS DeepLens, 2017
#

import os
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
from threading import Thread
from playsound import playsound
from botocore.session import Session
from recognitionobject import RecognitionObject
from validator import AttireValidator

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has 
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
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
        client.publish(topic=iotTopic, payload="Opened Pipe")
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
    try:
        modelPath = "/opt/awscam/artifacts/VGG_VOC0712_SSD_300x300_deploy.xml"
        modelType = "ssd"
        input_width = 300
        input_height = 300
        max_threshold = 0.5
        outMap = { 1: 'Jeans', 2: 'Tee', 3: 'Blazer', 4: 'Face', 5: 'Person' }
        results_thread = FIFO_Thread()
        results_thread.start()
        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Dress code recognition starts now")

        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload="Model loaded")
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")
            
        yscale = float(frame.shape[0]/input_height)
        xscale = float(frame.shape[1]/input_width)

        doInfer = True
        framesToSkip = 0
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            label = '{'
            if framesToSkip == 0:
               # setting infer mode ON
               cv2.putText(frame, "Infer Mode: ON", (0, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,128), 3)

               # Resize frame to fit model input requirement
               frameResize = cv2.resize(frame, (input_width, input_height))

               # Run model inference on the resized frame
	       inferOutput = model.doInference(frameResize)

               # Output inference result to the fifo file so it can be viewed with mplayer
               parsed_results = model.parseResult(modelType, inferOutput)['ssd']
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
               try: 
                  framesToSkip = makeFrameDecision(frameResize)
               except Exception as ex:
                  exception = "Received exception while making frame decision: " + str(ex)
                  client.publish(topic=iotTopic, payload=exception)
            else:
                framesToSkip -= 1
                # setting infer mode OFF
                cv2.putText(frame, "Infer Mode: OFF", (0, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,128), 3)
                    
            label += '"null": 0.0'
            label += '}' 
            client.publish(topic=iotTopic, payload = label)
            global jpeg
            ret,jpeg = cv2.imencode('.jpg', frame)
            
    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()
    
def writeFrameToS3(frame):
    session = Session()
    s3 = session.create_client('s3')
    file_name = 'unrecognized-images/image-'+time.strftime("%Y%m%d-%H%M%S")+'.jpg'
    ret, image = cv2.imencode('.jpg', frame)
    image_bytes = image.tobytes()
    response = s3.put_object(ACL='public-read', Body=image_bytes, Bucket='deeplens-dresscode-recognition-images', Key=file_name)

    s3Url = 'https://s3.amazonaws.com/deeplens-dresscode-recognition-images/'+file_name
    message = 'Manual dress code check needed. You can view the image at ' + s3Url
    sns = session.create_client('sns', region_name='us-east-1')
    snsResponse = sns.publish(TargetArn='arn:aws:sns:us-east-1:565813021316:Manual_Dresscode_Check_Needed', 
                              Message=message,
                              Subject='Manual dress check needed',
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
    if valid_frame == 7:
      resetFrameCounter()
      playsound("/opt/awscam/artifacts/access-granted.mp3")
      return 50

    elif invalid_frame == 7:
      resetFrameCounter()
      playsound("/opt/awscam/artifacts/access-denied.mp3")
      return 50

    elif manual_check_frame == 7:
      resetFrameCounter()
      playsound("/opt/awscam/artifacts/please-wait.mp3")
      writeFrameToS3(frame)
      return 50
    else:
      return 0

# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
