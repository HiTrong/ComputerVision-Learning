import cv2
import numpy as np
import streamlit as st
from PIL import Image

L = 256
#-----Function Chapter 5-----#
def CreateMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = T*np.cos(phi)
                IM = -T*np.sin(phi)
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H

def CreateMotionNoise(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float32)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateMotionfilter(M, N)

    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def CreateInverseMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DenoiseMotion(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateInverseMotionfilter(M, N)
    H = H.astype(np.complex128)
    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

#================================================modified ===========================================
def CreateMotionNoise_streamlit():
    st.markdown('''<h2 class="subheader-text">
                5.1/ Tạo nhiễu chuyển động
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Tạo nhiễu chuyển động (motion blur) trong xử lý ảnh là quá trình mô phỏng hiện tượng nhiễu xuất phát từ chuyển động của đối tượng hoặc máy ảnh trong quá trình chụp ảnh. Nhiễu chuyển động tạo ra một hiệu ứng mờ dọc theo hướng chuyển động, giống như là khi một đối tượng đang di chuyển nhanh trong khi ảnh được chụp.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt5_1, bt5_1_another,bt5_1_delete = st.columns(3)
    bt5_1_state = bt5_1.button("Xử lý demo",type="primary")
    bt5_1_another_state = bt5_1_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt5_1_state:
        cot1.image(path+"5_1.tif",use_column_width=True)
        img = cv2.imread(path+"5_1.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(CreateMotionNoise(img),use_column_width=True)
        bt5_1_delete_state = bt5_1_delete.button("Xóa",type="primary")
    if bt5_1_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(CreateMotionNoise(frame),use_column_width=True)
            bt5_1_delete_state = bt5_1_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def DenoiseMotion_streamlit():
    st.markdown('''<h2 class="subheader-text">
                5.2/ Gỡ nhiễu của ảnh có ít nhiễu
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Gỡ nhiễu của ảnh có ít nhiễu trong xử lý ảnh thường nhằm mục đích làm tăng chất lượng hình ảnh bằng cách giảm hoặc loại bỏ các thành phần nhiễu không mong muốn mà có thể xuất hiện trong quá trình chụp hình. ")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt5_2, bt5_2_another,bt5_2_delete = st.columns(3)
    bt5_2_state = bt5_2.button("Xử lý demo",type="primary")
    bt5_2_another_state = bt5_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt5_2_state:
        cot1.image(path+"5_2.png",use_column_width=True)
        img = cv2.imread(path+"5_2.png",cv2.IMREAD_GRAYSCALE)
        cot2.image(DenoiseMotion(img),use_column_width=True)
        bt5_2_delete_state = bt5_2_delete.button("Xóa",type="primary")
    if bt5_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(DenoiseMotion(frame),use_column_width=True)
            bt5_2_delete_state = bt5_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def DenoisestMotion_streamlit():
    st.markdown('''<h2 class="subheader-text">
                5.3/ Gỡ nhiễu của ảnh có nhiều nhiễu
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Gỡ nhiễu của ảnh có nhiều nhiễu trong xử lý ảnh là quá trình giảm hoặc loại bỏ các thành phần nhiễu từ hình ảnh khi ảnh chứa một lượng đáng kể nhiễu. Các ảnh có nhiều nhiễu thường gặp trong các điều kiện chụp ảnh yếu, ánh sáng thấp, hoặc khi sử dụng các thiết bị ảnh có độ nhạy nhiễu cao.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt5_3, bt5_3_another,bt5_3_delete = st.columns(3)
    bt5_3_state = bt5_3.button("Xử lý demo",type="primary")
    bt5_3_another_state = bt5_3_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt5_3_state:
        cot1.image(path+"5_3.tif",use_column_width=True)
        img = cv2.imread(path+"5_3.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(DenoiseMotion(cv2.medianBlur(img,7)),use_column_width=True)
        bt5_3_delete_state = bt5_3_delete.button("Xóa",type="primary")
    if bt5_3_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(DenoiseMotion(cv2.medianBlur(frame,7)),use_column_width=True)
            bt5_3_delete_state = bt5_3_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
    
# File ảnh
path = ".\\Module\\XuLyAnh\\img\\chuong5\\"

def Chuong5_streamlit():
    st.markdown('''<h2 class="title">
                Chương 5: Khôi phục ảnh
                </h2>
                ''',unsafe_allow_html=True)
    
    tab_names = ["Tạo nhiễu chuyển động", "Gỡ nhiễu của ảnh có ít nhiễu",
                 "Gỡ nhiễu của ảnh có nhiều nhiễu"]
    selected_tab = st.sidebar.radio("Chọn nội dung xử lý:", tab_names)
    
    if selected_tab =="Tạo nhiễu chuyển động":
        CreateMotionNoise_streamlit()
    if selected_tab == "Gỡ nhiễu của ảnh có ít nhiễu":
        DenoiseMotion_streamlit()
    if selected_tab == "Gỡ nhiễu của ảnh có nhiều nhiễu":
        DenoisestMotion_streamlit()
    

    
