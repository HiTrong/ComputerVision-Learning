# Import required modules
import cv2 as cv
import time
import argparse
import streamlit as st
import numpy as np
import time
import os
import shutil
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from ffmpy import FFmpeg

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


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                         (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

faceProto = "Module/FaceAgeGenderDectected/model/opencv_face_detector.pbtxt"
faceModel = "Module/FaceAgeGenderDectected/model/opencv_face_detector_uint8.pb"

ageProto = "Module/FaceAgeGenderDectected/model/age_deploy.prototxt"
ageModel = "Module/FaceAgeGenderDectected/model/age_net.caffemodel"

genderProto = "Module/FaceAgeGenderDectected/model/gender_deploy.prototxt"
genderModel = "Module/FaceAgeGenderDectected/model/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

def run(frame):
    frameFace, bboxes = getFaceBox(faceNet, frame)
    padding = 15
    if not bboxes:
        return frame

    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1),
                     max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        if face is not None:
            blob = cv.dnn.blobFromImage(
                face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        # print("Gender Output : {}".format(genderPreds))
        #print("Gender : {}, conf = {:.3f}".format(
            #gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        #print("Age Output : {}".format(agePreds))
        #print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    return frameFace




        


# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv_gpu -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/cv2_gpu/opencv_contrib/modules -DPYTHON_EXECUTABLE=~/env/bin/python3 -DBUILD_EXAMPLES=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON  -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DOpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ ..

# ======================================= streamlit ============================================
def FaceAgeGenderDectected_streamlit_show():
    result_path = ".\\Module\\FaceAgeGenderDectected\\result\\"
    
    # Title
    st.markdown('<h1 class="title">Nhận dạng gương mặt, giới tính và độ tuổi</h1>', unsafe_allow_html=True)

    # Logo
    st.image("GUI/img/facegenderage_detect.jpg")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model để nhận dạng khuôn mặt cùng với độ tuổi với giới tính của các đối tượng 
                <h2 class="subheader-text">=> Hãy upload ảnh hoặc video bạn muốn chúng tôi nhận dạng ở bên dưới nhé! Ngoài ra bạn cũng có thể mở webcam để nhận dạng (Khuyến cáo máy mạnh)!</h2>
                <h2 class="subheader-text warning">Lưu ý: Hãy click button xóa để dọn sạch vùng nhớ tạm thời và kết quả khi không cần sử dụng nữa nhé!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
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

    if bt3:
        output_file = ".\\Module\\FaceAgeGenderDectected\\video\\"
        if os.path.exists(output_file):
            shutil.rmtree(output_file)
    else:
        if bt1:
            with st.status("Start face with age and gender detect", expanded=True) as status:
                st.write("Opening WebCam")
                bt_stop = st.button("Dừng lại",type="primary")
                place_holder = st.image([])
                    
                cap = cv2.VideoCapture(0)
                while (cap.isOpened() and not bt_stop):
                    ret, frame =cap.read()
                    img = run(frame)
                    
                    # Hiển thị lên streamlit
                    place_holder.image(img,channels="BGR")
                    if not ret:
                        break
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or bt_stop:
                        break
                cap.release()
        else:
            # Xử lý khi đã upload file
            if uploaded_file is not None:
                if "image" in uploaded_file.type:
                    st.image(uploaded_file)
                    if bt2:
                        with st.status("Start face with age and gender detect", expanded=True) as status:
                            st.write("Detecting Image ...")
                            image = Image.open(uploaded_file)
                            frame = np.array(image)
                            frame = frame[:, :, [2, 1, 0]] 
                            img = run(frame)
                            st.write("Detect done")
                            st.image(img, caption=None, channels="BGR")
                
                if "video" in uploaded_file.type:
                    output_file = ".\\Module\\FaceAgeGenderDectected\\video\\"
                    
                    # Lưu file upload
                    os.makedirs(output_file,exist_ok=True)
                    dest_path = output_file+ uploaded_file.name
                    with open(dest_path, "wb") as dest_file:
                        dest_file.write(uploaded_file.read())
                    
                    st.video(uploaded_file)
                    if bt2:
                        with st.status("Start face with age and gender detect", expanded=True) as status:
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
                                
                                img = run(frame)
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

