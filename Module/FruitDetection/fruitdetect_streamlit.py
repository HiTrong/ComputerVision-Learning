import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import os
import shutil
from Module.FruitDetection.detect import run
from moviepy.editor import VideoFileClip
from ffmpy import FFmpeg

result_path = ".\\Module\\FruitDetection\\runs\\detect\\result\\"
weight_path = ".\\Module\\FruitDetection\\best.onnx"

def frame_detect(uploaded_file):
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    image = Image.open(uploaded_file)
    input_filepath = ".\\Module\\FruitDetection\\input.png"
    image.save(input_filepath)
    
    
    run(weights=weight_path, source=input_filepath, name="result",conf_thres=0.48)
    
def video_detect(uploaded_file):
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    # lưu video
    video_bytes = uploaded_file.read()
    input_filepath = ".\\Module\\FruitDetection\\input.mp4"
    with open(input_filepath, 'wb') as file:
        file.write(video_bytes)
        
    run(weights=weight_path, source=input_filepath, name="result",conf_thres=0.48)
        
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

def fruitdetect_streamlit_show():
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
        if os.path.exists(".\\Module\\FruitDetection\\input.mp4"):
            os.remove(".\\Module\\FruitDetection\\input.mp4")
        if os.path.exists(".\\Module\\FruitDetection\\input.png"):
            os.remove(".\\Module\\FruitDetection\\input.png")
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
                    with st.status("Fruit Detection Start!", expanded=True) as status:
                        st.write("Open image: "+uploaded_file.name)
                        st.write("Detect...")
                        frame_detect(uploaded_file)
                        image = Image.open(result_path + "input.png")
                        st.write("Done!")
                        st.image(image)
            if "video" in uploaded_file.type:
                st.video(uploaded_file) 
                if bt1:
                    with st.status("Fruit Detection Start!", expanded=True) as status:
                        st.write("Open video: "+uploaded_file.name)
                        st.write("Detect...")
                        video_detect(uploaded_file)
                        st.write("Detect done!")
                        st.write("Start convert video!")
                        convert_video(input_path=result_path + "input.mp4",
                                output_path=result_path + 'output.mp4')
                        st.write("All done!")
                        st.video(result_path + 'output.mp4')
            






