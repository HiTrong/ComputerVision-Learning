import numpy as np
import cv2
import streamlit as st
from PIL import Image
L = 256

def Spectrum(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 và phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin
    fp = fp/(L-1)

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Tính spectrum
    S = np.sqrt(F[:,:,0]**2 + F[:,:,1]**2)
    S = np.clip(S, 0, L-1)
    S = S.astype(np.uint8)
    return S

def FrequencyFilter(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: 
    # Tạo bộ lọc H thực High Pass Butterworth
    H = np.zeros((P,Q), np.float32)
    D0 = 60
    n = 2
    for u in range(0, P):
        for v in range(0, Q):
            Duv = np.sqrt((u-P//2)**2 + (v-Q//2)**2)
            if Duv > 0:
                H[u,v] = 1.0/(1.0 + np.power(D0/Duv,2*n))
    # Bước 6:
    # G = F*H nhân từng cặp
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    
    # Bước 7:
    # IDFT
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    # Lấy phần thực
    gp = g[:,:,0]
    # Nhân với (-1)^(x+y)
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    # Bước 8:
    # Lấy kích thước ảnh ban đầu
    imgout = gp[0:M,0:N]
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def CreateNotchRejectFilter():
    P = 250
    Q = 180
    u1, v1 = 44, 58
    u2, v2 = 40, 119
    u3, v3 = 86, 59
    u4, v4 = 82, 119

    D0 = 10
    n = 2
    H = np.ones((P,Q), np.float32)
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            # Bộ lọc u1, v1
            Duv = np.sqrt((u-u1)**2 + (v-v1)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u1))**2 + (v-(Q-v1))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u2, v2
            Duv = np.sqrt((u-u2)**2 + (v-v2)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u2))**2 + (v-(Q-v2))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u3, v3
            Duv = np.sqrt((u-u3)**2 + (v-v3)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u3))**2 + (v-(Q-v3))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0

            # Bộ lọc u4, v4
            Duv = np.sqrt((u-u4)**2 + (v-v4)**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            Duv = np.sqrt((u-(P-u4))**2 + (v-(Q-v4))**2)
            if Duv > 0:
                h = h*1.0/(1.0 + np.power(D0/Duv,2*n))
            else:
                h = h*0.0
            H[u,v] = h
    return H

def DrawNotchRejectFilter():
    H = CreateNotchRejectFilter()
    H = H*(L-1)
    H = H.astype(np.uint8)
    return H
    
def RemoveMoire(imgin):
    M, N = imgin.shape
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    
    # Bước 1 và 2: 
    # Tạo ảnh mới có kích thước PxQ
    # và thêm số 0 vào phần mở rộng
    fp = np.zeros((P,Q), np.float32)
    fp[:M,:N] = imgin

    # Bước 3:
    # Nhân (-1)^(x+y) để dời vào tâm ảnh
    for x in range(0, M):
        for y in range(0, N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    # Bước 4:
    # Tính DFT    
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Bước 5: 
    # Tạo bộ lọc NotchReject 
    H = CreateNotchRejectFilter()
    # Bước 6:
    # G = F*H nhân từng cặp
    G = F.copy()
    for u in range(0, P):
        for v in range(0, Q):
            G[u,v,0] = F[u,v,0]*H[u,v]
            G[u,v,1] = F[u,v,1]*H[u,v]
    
    # Bước 7:
    # IDFT
    g = cv2.idft(G, flags = cv2.DFT_SCALE)
    # Lấy phần thực
    gp = g[:,:,0]
    # Nhân với (-1)^(x+y)
    for x in range(0, P):
        for y in range(0, Q):
            if (x+y)%2 == 1:
                gp[x,y] = -gp[x,y]
    # Bước 8:
    # Lấy kích thước ảnh ban đầu
    imgout = gp[0:M,0:N]
    imgout = np.clip(imgout,0,L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Spectrum_streamlit():
    st.markdown('''<h2 class="subheader-text">
                4.1/ Spectrum
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Xử lý quang phổ bằng các công thức tần số sóng điện tử, ánh sáng, điểm ảnh.")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt4_1, bt4_1_another,bt4_1_delete = st.columns(3)
    bt4_1_state = bt4_1.button("Xử lý demo",type="primary")
    bt4_1_another_state = bt4_1_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt4_1_state:
        cot1.image(path+"4_1.tif",use_column_width=True)
        img = cv2.imread(path+"4_1.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Spectrum(img),use_column_width=True)
        bt4_1_delete_state = bt4_1_delete.button("Xóa",type="primary")
    if bt4_1_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Spectrum(frame),use_column_width=True)
            bt4_1_delete_state = bt4_1_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def Highpass_streamlit():
    st.markdown('''<h2 class="subheader-text">
                4.2/ Lọc trong miền tần số - highpass filter
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Bộ lọc cho tín hiệu có tần số thấp hơn tần số cắt đã chọn và làm suy giảm tín hiệu có tần số cao hơn tần số cắt để xử lý ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt4_2, bt4_2_another,bt4_2_delete = st.columns(3)
    bt4_2_state = bt4_2.button("Xử lý demo",type="primary")
    bt4_2_another_state = bt4_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt4_2_state:
        cot1.image(path+"4_2.tif",use_column_width=True)
        img = cv2.imread(path+"4_2.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(FrequencyFilter(img),use_column_width=True)
        bt4_2_delete_state = bt4_2_delete.button("Xóa",type="primary")
    if bt4_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(FrequencyFilter(frame),use_column_width=True)
            bt4_2_delete_state = bt4_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
    
def DrawNotchRejectFilter_streamlit():
    st.markdown('''<h2 class="subheader-text">
                4.3.1/ Vẽ bộ lọc Notch Reject
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để xử lý lọc ảnh cơ bản.")
    bt4_3_1, bt4_3_1_delete = st.columns(2)
    bt4_3_1_state = bt4_3_1.button("Xử lý Vẽ bộ lọc Notch Reject",type="primary")
    if bt4_3_1_state:
        st.image(DrawNotchRejectFilter())
        bt4_3_1_delete.button("Xóa",type="primary")

def RemoveMoire_streamlit():
    st.markdown('''<h2 class="subheader-text">
                4.3.2/ Xóa nhiễu moire
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để xóa các điểm nhiễu ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt4_3_2, bt4_3_2_another,bt4_3_2_delete = st.columns(3)
    bt4_3_2_state = bt4_3_2.button("Xử lý demo",type="primary")
    bt4_3_2_another_state = bt4_3_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt4_3_2_state:
        cot1.image(path+"4_3_2.tif",use_column_width=True)
        img = cv2.imread(path+"4_3_2.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(RemoveMoire(img),use_column_width=True)
        bt4_3_2_delete_state = bt4_3_2_delete.button("Xóa",type="primary")
    if bt4_3_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(RemoveMoire(frame),use_column_width=True)
            bt4_3_2_delete_state = bt4_3_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
    

# File ảnh
path = ".\\Module\\XuLyAnh\\img\\chuong4\\"

def Chuong4_streamlit():
    st.markdown('''<h2 class="title">
                Chương 4: Lọc trong miền tần số
                </h2>
                ''',unsafe_allow_html=True)
    
    tab_names = ["Spectrum","Lọc trong miền tần số",
                 "Vẽ bộ lọc Notch Reject","Xóa nhiễu moire"]
    selected_tab = st.sidebar.radio("Chọn nội dung xử lý:", tab_names)
    
    if selected_tab == "Spectrum":
        Spectrum_streamlit()
    if selected_tab == "Lọc trong miền tần số":
        Highpass_streamlit()
    if selected_tab == "Vẽ bộ lọc Notch Reject":
        DrawNotchRejectFilter_streamlit()
    if selected_tab == "Xóa nhiễu moire":
        RemoveMoire_streamlit()
    
