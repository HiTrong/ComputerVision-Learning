import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import shutil
from streamlit_drawable_canvas import st_canvas
from PIL import Image

L = 256

def Erosion(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(45,45))
    cv2.erode(imgin,w,imgout)

def Dilation(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    cv2.dilate(imgin,w,imgout)
def OpeningClosing(imgin, imgout):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_OPEN, w)
    cv2.morphologyEx(temp, cv2.MORPH_CLOSE, w, imgout)

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    temp = cv2.erode(imgin,w)
    imgout = imgin - temp
    return imgout

def HoleFill(imgin):
    imgout = imgin
    M, N = imgout.shape
    mask = np.zeros((M+2,N+2),np.uint8)
    cv2.floodFill(imgout,mask,(105,297),L-1)
    return imgout

def MyConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    M, N = temp.shape
    dem = 0
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            if temp[x,y] == L-1:
                mask = np.zeros((M+2,N+2),np.uint8)
                cv2.floodFill(temp, mask, (y,x), (color,color,color))
                dem = dem + 1
                color = color + 1
    print('Co %d thanh phan lien thong' % dem)
    a = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = temp[x,y]
            if r > 0:
                a[r] = a[r] + 1
    dem = 1
    for r in range(0, L):
        if a[r] > 0:
            print('%4d   %5d' % (dem, a[r]))
            dem = dem + 1
    return temp

def ConnectedComponent(imgin):
    ret, temp = cv2.threshold(imgin, 200, L-1, cv2.THRESH_BINARY)
    temp = cv2.medianBlur(temp, 7)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d thanh phan lien thong' % (dem-1) 
    print(text)

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(1, dem):
        print('%4d %10d' % (r, a[r]))
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label

def CountRice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = 'Co %d hat gao' % (dem-1) 
    print(text)
    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    for r in range(0, dem):
        print('%4d %10d' % (r, a[r]))

    max = a[1]
    rmax = 1
    for r in range(2, dem):
        if a[r] > max:
            max = a[r]
            rmax = r

    xoa = np.array([], np.int32)
    for r in range(1, dem):
        if a[r] < 0.5*max:
            xoa = np.append(xoa, r)

    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                r = r - color
                if r in xoa:
                    label[x,y] = 0
    label = label.astype(np.uint8)
    cv2.putText(label,text,(1,25),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255,255,255),2)
    return label


# =======================================Modified ==========================================
def ConnectedComponent_streamlit():
    st.markdown('''<h2 class="subheader-text">
                9.1/ Đếm thành phần liên thông của miếng phi lê gà (Connected Component)
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để đếm thành phần liên thông và tách ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt9_1, bt9_1_another,bt9_1_delete = st.columns(3)
    bt9_1_state = bt9_1.button("Xử lý demo",type="primary")
    bt9_1_another_state = bt9_1_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt9_1_state:
        cot1.image(path+"9_1.tif",use_column_width=True)
        img = cv2.imread(path+"9_1.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(ConnectedComponent(img),use_column_width=True)
        bt9_1_delete_state = bt9_1_delete.button("Xóa",type="primary")
    if bt9_1_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(ConnectedComponent(frame),use_column_width=True)
            bt9_1_delete_state = bt9_1_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def CountRice_streamlit():
    st.markdown('''<h2 class="subheader-text">
                9.2/ Đếm hạt gạo (Count Rice)
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để đếm thành phần trong ảnh")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt9_2, bt9_2_another,bt9_2_delete = st.columns(3)
    bt9_2_state = bt9_2.button("Xử lý demo",type="primary")
    bt9_2_another_state = bt9_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt9_2_state:
        cot1.image(path+"9_2.tif",use_column_width=True)
        img = cv2.imread(path+"9_2.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(CountRice(img),use_column_width=True)
        bt9_2_delete_state = bt9_2_delete.button("Xóa",type="primary")
    if bt9_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(CountRice(frame),use_column_width=True)
            bt9_2_delete_state = bt9_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def HoleFillingMouse_streamlit():
    st.markdown('''<h2 class="subheader-text">
                9.3/ Hole Filling Mouse
                </h2>
                ''',unsafe_allow_html=True)
    # Chèn đoạn mã JavaScript để theo dõi sự kiện click trên canvas
    components.html(
        """
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');

            canvas.addEventListener('click', (e) => {
                const x = e.clientX - canvas.getBoundingClientRect().left;
                const y = e.clientY - canvas.getBoundingClientRect().top;
                console.log(`Clicked at (${x}, ${y})`);

                // Truyền tọa độ chuột về Streamlit sử dụng Streamlit's JS API
                Streamlit.setComponentValue({x, y});
            });
        </script>
        """,
        height=0,
    )
    
    st.write("Click vào hole muốn fill!")
    bt_reset = st.button("Reset!",type="primary")
    img = Image.open(path+"9_3_source.tif")
    col1,col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(background_image= img,width=512, height=512,fill_color="rgb(0, 255, 255)")
    with col2:
        st.image(path+"9_3\\9_3.tif",caption="Ảnh gốc")
    def loop():
        img_cv2 = cv2.imread(path+"9_3_source.tif",cv2.IMREAD_GRAYSCALE)
        img = Image.open(path+"9_3_source.tif")
        if canvas_result.json_data:
            if canvas_result.json_data.get("objects"):
                coordinates = (0,0)
                for obj in canvas_result.json_data.get("objects", []):
                    # Lấy tọa độ từ thuộc tính `left` và `top` của mỗi đối tượng
                    coordinates = (obj.get("left", 0), obj.get("top", 0))
                M,N = img_cv2.shape
                cv2.floodFill(img_cv2, np.zeros((M+2, N+2), np.uint8), (int(coordinates[0]),int(coordinates[1])), (255,255,255))
                cv2.imwrite(path+"9_3_source.tif",img_cv2)
                canvas_result.image_data = Image.open(path + "9_3_source.tif").copy()
        else:
            canvas_result.image_data = Image.open(path+"9_3_source.tif").copy()
            
    if bt_reset:
        shutil.copy(path+"9_3\\9_3.tif",path+"9_3_source.tif")
        loop()
    else:
        loop()

# File ảnh
path = ".\\Module\\XuLyAnh\\img\\chuong9\\"

def Chuong9_streamlit():
    st.markdown('''<h2 class="title">
                Chương 9: Xử lý ảnh hình thái
                </h2>
                ''',unsafe_allow_html=True)
    
    tab_names = ["Connected Component",
                 "Đếm hạt gạo (Count Rice)","Hole Filling Mouse"]
    selected_tab = st.sidebar.radio("Chọn nội dung xử lý:", tab_names)
    
    if selected_tab == "Connected Component":
        ConnectedComponent_streamlit()
    if selected_tab == "Đếm hạt gạo (Count Rice)":
        CountRice_streamlit()
    if selected_tab == "Hole Filling Mouse":
        HoleFillingMouse_streamlit()