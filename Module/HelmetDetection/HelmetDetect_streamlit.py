from ultralytics import YOLO
import cv2
import math
import streamlit as st
import time
import shutil
import os
import numpy as np
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


model_helmet = YOLO("./Module/HelmetDetection/weights/best.pt")



def detect_helmet_video(video_path,result_path):
    os.makedirs(result_path,exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # Lấy thông số của video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Tạo đối tượng VideoWriter
    out = cv2.VideoWriter(result_path + "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    frame_count = 0
    
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Gửi frame vào hàm nhận diện
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = detect_helmet_frame(frame)

        # Ghi frame đã được xử lý vào video output
        out.write(frame)
        if (frame_count % 10 == 0):
            st.write("Frame Recognized: "+str(frame_count))
    cap.release()
    out.release()
    st.write("Frame Recognized: "+str(frame_count-1))
    
    
def detect_helmet_frame(frame):
    result_string = []
    
    helmets_detected = model_helmet(frame,stream=True)
    
    for helmet in helmets_detected:
        boxes = helmet.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            
            # In lên frame
            org = [x1, y1-3]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (0, 0, 255)
            thickness = 2
            cv2.putText(frame, "Mu Bao Hiem", org, font, fontScale, color, thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    return frame

                    
# ==================================== streamlit =======================================

def HelmetDectected_streamlit_show():
    result_path = ".\\Module\\HelmetDetection\\result\\"
    
    # Title
    st.markdown('<h1 class="title">Nhận dạng mũ bảo hiểm Yolo v8 (Custom)</h1>', unsafe_allow_html=True)

    # Logo fruit
    st.image("GUI/img/helmetdetection.png")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model tự train với dữ liệu custom về mũ bảo hiểm để nhận dạng các mũ bảo hiểm. Kết quả chúng ta sẽ nhận dạng được liệu người đi đường có đội mũ bảo hiểm hay không! 
                <h2 class="subheader-text">=> Hãy upload ảnh hoặc video bạn muốn chúng tôi nhận dạng ở bên dưới nhé!</h2>
                <h2 class="subheader-text warning">Lưu ý: Hãy click button xóa để dọn sạch vùng nhớ tạm thời và kết quả khi không cần sử dụng nữa nhé!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, MP4 file",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Scanned file are not supported yet!",
    )
    
    col1, col2 = st.columns(2)
    bt1 = col1.button("Nhận dạng",type="primary")
    bt2 = col2.button("Xóa",type="primary")
      
    if bt2:
        if os.path.exists(".\\Module\\HelmetDetection\\input.mp4"):
            os.remove(".\\Module\\HelmetDetection\\input.mp4")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
    else:
        if not uploaded_file:
            st.warning("Vui lòng upload file để nhận dạng!!!")
        else:
            with st.spinner('Wait for it...'):    
                time.sleep(1)
                st.progress(100)
            if "image" in uploaded_file.type:
                st.image(uploaded_file)
                if bt1:
                    with st.status("Helmet Detection Start!", expanded=True) as status:
                        st.write("Open image: "+uploaded_file.name)
                        image = Image.open(uploaded_file)
                        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        st.write("Detecting...")
                        image = detect_helmet_frame(frame)
                        st.write("Done!")
                        status.update(label= "Completed Helmet Detection",state="complete", expanded=False)
                    st.image(image,"Ảnh đã nhận diện (Để biết thêm chi tiết, click vào status)",channels="BGR")
                
            if "video" in uploaded_file.type:
                st.video(uploaded_file) 
                if bt1:
                    with st.status("Helmet Detection Start!", expanded=True) as status:
                        st.write("Open video: "+uploaded_file.name)
                        if os.path.exists(result_path):
                            shutil.rmtree(result_path)
                        # lưu video
                        video_bytes = uploaded_file.read()
                        input_filepath = ".\\Module\\HelmetDetection\\input.mp4"
                        with open(input_filepath, 'wb') as file:
                            file.write(video_bytes)

                        st.write("Detecting...")
                        detect_helmet_video(input_filepath,result_path)
                        st.write("Detection done!")
                        st.write("Start convert video!")
                        convert_video(input_path=result_path + "output.mp4",
                                output_path=result_path + 'output_convert.mp4')
                        st.write("All done!")
                        status.update(label= "Detection successfully! Video is available!",state="complete", expanded=False)
                    st.video(result_path + 'output_convert.mp4')