import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time
import os
from moviepy.editor import VideoFileClip
from ffmpy import FFmpeg
import shutil


classes = None
with open('.\\Module\\ObjectDetection\\object_detection_classes_yolov4.txt', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

try:
      if st.session_state["LoadModel4"] == True:
            print('Đã load model')
            pass
except:
      st.session_state["LoadModel4"] = True
      st.session_state["Net4"] = cv2.dnn.readNet('.\\Module\\ObjectDetection\\yolov4.weights', '.\\Module\\ObjectDetection\\yolov4.cfg')
      print('Load model lần đầu')
st.session_state["Net4"].setPreferableBackend(0)
st.session_state["Net4"].setPreferableTarget(0)
outNames = st.session_state["Net4"].getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4

# Convert video
def convert_video(input_path, output_path):
    codec='libx264' 
    audio_codec='aac'
    try:
        video_clip = VideoFileClip(input_path)
        video_clip.write_videofile(output_path, codec=codec, audio_codec=audio_codec)
        return True
    except Exception as e:
        print(f"Lỗi khi chuyển đổi video: {e}")
        return False




def postprocess(frame, outs):
    frame = frame.copy()
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = st.session_state["Net4"].getLayerNames()
    lastLayerId = st.session_state["Net4"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net4"].getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # NMS is used inside Region layer only on DNN_BACKEND_OPENcv2 for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or lastLayer.type == 'Region' and 0 != cv2.dnn.DNN_BACKEND_OPENcv2:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            nms_indices = nms_indices[:] if len(nms_indices) else []
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return frame


# ================================== Code modified ==================================
def frame_detect(frame):
    inpWidth = 416
    inpHeight = 416
    blob = cv2.dnn.blobFromImage(frame.copy(), size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)
    # Run a model
    st.session_state["Net4"].setInput(blob, scalefactor=0.00392, mean=[0, 0, 0])
    outs = st.session_state["Net4"].forward(outNames)
    img = postprocess(frame, outs)   
    return img    

def streamlit_yolov4():
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, MP4 file",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Scanned file are not supported yet!",
    )    
    
    
    col1, col2, col3 = st.columns(3)
    bt1 = col1.button("Nhận dạng qua webcam",type="primary")
    bt2 = col2.button("Nhận dạng file upload",type="primary")
    bt3 = col3.button("Xóa",type="primary")

    if bt1:
        with st.status("Start object detect", expanded=True) as status:
            st.write("Opening WebCam")
            bt_stop = st.button("Dừng lại",type="primary")
            place_holder = st.image([])
                
            cap = cv2.VideoCapture(0)
            while (cap.isOpened() and not bt_stop):
                ret, frame =cap.read()
                img = frame_detect(frame)
                
                # Hiển thị lên streamlit
                place_holder.image(img,channels="BGR")
                if not ret:
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or bt_stop:
                    break
            cap.release()
    
    if bt3:
        output_file = ".\\Module\\ObjectDetection\\video\\"
        if os.path.exists(output_file):
            shutil.rmtree(output_file)
    else:
        # Xử lý khi đã upload file
        if uploaded_file is not None:
            if "image" in uploaded_file.type:
                st.image(uploaded_file)
                if bt2:
                    with st.status("Detect Object Yolov4", expanded=True) as status:
                        st.write("Detecting Image ...")
                        image = Image.open(uploaded_file)
                        frame = np.array(image)
                        frame = frame[:, :, [2, 1, 0]] 
                        img = frame_detect(frame)
                        st.write("Detect done")
                        st.image(img, caption=None, channels="BGR")
                
                
            if "video" in uploaded_file.type:
                st.video(uploaded_file)
                output_file = ".\\Module\\ObjectDetection\\video\\"
                if bt2: 
                    # Lưu file upload
                    os.makedirs(output_file,exist_ok=True)
                    dest_path = output_file+ uploaded_file.name
                    with open(dest_path, "wb") as dest_file:
                        dest_file.write(uploaded_file.read())
                    
                    # st.video(uploaded_file)

                    with st.status("Detect Object Yolov4", expanded=True) as status:
                        st.write("Detecting Video ...")
                        # Đọc video sử dụng OpenCV
                        cap = cv2.VideoCapture(dest_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = cap.get(3)
                        height = cap.get(4)
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(output_file + "output.mp4", fourcc, fps, (int(width), int(height)))
                        frame_count = 0
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            if (frame_count % 10) == 0:
                                st.write("Frame Detected: ", frame_count)
                            
                            img = frame_detect(frame)
                            out.write(img)
                        
                        cap.release()
                        out.release()
                        
                        st.write("Detect video is done!!!")
                        time.sleep(1)
                        st.write("Convert and save video ...")
                        convert_video(input_path=output_file+"output.mp4",
                                    output_path=output_file+"converted_output.mp4")
                        st.write("Video đã được lưu tại: "+ output_file + "converted_output.mp4")
                        status.update(label= "Successfully! Video is available!",state="complete", expanded=False)
                    st.video(output_file + "converted_output.mp4")
    
    st.stop()