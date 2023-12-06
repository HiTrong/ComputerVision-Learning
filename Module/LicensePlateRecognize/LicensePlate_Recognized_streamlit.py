from ultralytics import YOLO
import cv2
import math
import time
import numpy as np
import os
import shutil
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



plate_classes = ["Biển"]
number_plate_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L',
           'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']


model_plate = YOLO("./Module/LicensePlateRecognize/NhanDienBienSo/weights/best.pt")
model_numberplate = YOLO("./Module/LicensePlateRecognize/NhanDienChuSo/weights/best.pt")



def detect_video(video_path,result_path):
    os.makedirs(result_path,exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    # Lấy thông số của video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Tạo đối tượng VideoWriter
    out = cv2.VideoWriter(result_path + "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    #notice = []
    frame_count = 0
    
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Gửi frame vào hàm nhận diện
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        str_result, frame = detect_frame(frame)

        # Ghi frame đã được xử lý vào video output
        out.write(frame)
        if (frame_count % 10 == 0):
            st.write("Frame Recognized: "+str(frame_count))
    cap.release()
    out.release()
    st.write("Frame Recognized: "+str(frame_count-1))
    #return notice
    
def detect_frame(frame):
    result_string = []
    # nhận dạng biển số trước
    plates_detected = model_plate(frame,stream=True)
    
    for plate in plates_detected:
        boxes = plate.boxes
        
        for box in boxes:
            # Lấy ra tọa độ của các box biển số
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            
            # Cắt ảnh và nhận diện chữ và số có trong biển số
            frame_cut = frame[y1:y2,x1:x2]
            characters_detected = model_numberplate(frame_cut)

            list_x1 =[]
            list_y1 =[]
            list_x2 =[]
            list_y2 =[]
            list_char=[]
            for charac in characters_detected:
                boxes_char =  charac.boxes
                for box_char in boxes_char:
                    x1_char, y1_char, x2_char, y2_char = box_char.xyxy[0]
                    x1_char, y1_char, x2_char, y2_char = int(x1_char), int(y1_char), int(x2_char), int(y2_char)
                    list_x1.append(x1_char)
                    list_y1.append(y1_char)
                    list_x2.append(x2_char)
                    list_y2.append(y2_char)
                    list_char.append(number_plate_classes[int(box_char.cls[0])])
                  
                  
            # sắp xếp chữ số trên biển xe
            number_plate = sort_characters(list_x1,list_y1,list_x2,list_y2,list_char)        
            
            # In số lên trên biển xe
            org = [x1, y1-5]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.6
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame, number_plate, org, font, fontScale, color, thickness)
            result_string.append("Tìm thấy biển số: " + number_plate if len(number_plate)>0 else "Không thấy rõ số!")
    return result_string,frame
    
def sort_characters(x1,y1,x2,y2,char):
    # Sắp xếp theo x1 trước
    for i in range(0,len(x1)-1):
        for j in range(i+1,len(x1)):
            if x1[j] < x1[i]:
                x1[j],x1[i] = x1[i],x1[j]
                y1[j],y1[i] = y1[i],y1[j]
                x2[j],x2[i] = x2[i],x2[j]
                y2[j],y2[i] = y2[i],y2[j]
                char[j],char[i] = char[i],char[j]
    
    # Vì biển số xe có hai dạng: 1 hàng và 2 hàng nên ta sẽ tách nó ra
    length1 = 0
    length2 = 0
    first_char = []
    second_char = []
    
    for i in range(0,len(x1)-1):
        for j in range(i+1,len(x1)):
            check =0
            length = y1[j] + ((y2[j] - y1[j]) * 1/2) 
            if y1[i] > length and y2[i] > length:
                second_char.append(char[i])
                length2 = length if length2 == 0 else length2    
                check = 1
            else:
                if (y2[i] > length):
                    if (i != len(x1)-2):
                        continue
                    else:
                        if (y1[i] >length1): first_char.append(char[i]) 
                        else: second_char.append(char[i])
                        if (y1[j] >length1): first_char.append(char[j]) 
                        else: second_char.append(char[j])
                        break
                else:
                    first_char.append(char[i])
                    length1 = length if length1 == 0 else length1
                    check = 2           
            if (i != len(x1)-2):
                break
            else:
                if check == 1:
                    first_char.append(char[j])
                    length1 = length if length1 == 0 else length1
                else:
                    second_char.append(char[j])
                    length2 = length if length2 == 0 else length2    
                        
    if len(second_char)>0:
        return ''.join(first_char) + ''.join(second_char)
    else:
        return ''.join(char)





# ====================== Streamlit show ===============================
import streamlit as st



def LicensePlate_Recognized_streamlit_show():
    result_path = ".\\Module\\LicensePlateRecognize\\result\\"
    # Title
    st.markdown('<h1 class="title">Nhận diện biển số xe Yolo v8 (Custom)</h1>', unsafe_allow_html=True)

    # Logo fruit
    st.image("GUI/img/licenseplaterecognize.png")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model tự train với dữ liệu custom để nhận dạng các biển số xe và một model tự train với dữ liệu custom để nhận diện các chữ số trên biển số xe đã nhận dạng ở model trước đó. Kết quả chúng ta sẽ nhận diện được biển số xe! 
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
        if os.path.exists(".\\Module\\LicensePlateRecognize\\input.mp4"):
            os.remove(".\\Module\\LicensePlateRecognize\\input.mp4")
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
                    with st.status("Licesen Plate Recognization Start!", expanded=True) as status:
                        st.write("Open image: "+uploaded_file.name)
                        image = Image.open(uploaded_file)
                        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        st.write("Recognizing...")
                        result_string, image = detect_frame(frame)
                        st.write("Có " + str(len(result_string)) + " biển số ở trong hình, lịch sử nhận diện: ")
                        for s in result_string:
                            st.write(s)
                            time.sleep(0.5)
                        st.write("Done!")
                        status.update(label= "Completed recognize license plate",state="complete", expanded=False)
                    st.image(image,"Ảnh đã nhận diện (Để biết thêm chi tiết, click vào status)",channels="BGR")
                
            if "video" in uploaded_file.type:
                st.video(uploaded_file) 
                if bt1:
                    with st.status("Licesen Plate Recognization Start!", expanded=True) as status:
                        st.write("Open video: "+uploaded_file.name)
                        if os.path.exists(result_path):
                            shutil.rmtree(result_path)
                        # lưu video
                        video_bytes = uploaded_file.read()
                        input_filepath = ".\\Module\\LicensePlateRecognize\\input.mp4"
                        with open(input_filepath, 'wb') as file:
                            file.write(video_bytes)

                        st.write("Recognizing...")
                        detect_video(input_filepath,result_path)
                        st.write("Recognization done!")
                        st.write("Start convert video!")
                        convert_video(input_path=result_path + "output.mp4",
                                output_path=result_path + 'output_convert.mp4')
                        st.write("All done!")
                        status.update(label= "Recognization successfully! Video is available!",state="complete", expanded=False)
                    st.video(result_path + 'output_convert.mp4')