import numpy as np
import cv2
import streamlit as st
from PIL import Image

L = 256

def Negative(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = L-1-r
            imgout[x,y] = s
    return imgout

def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.log(1+r)
            imgout[x,y] = np.uint8(s)
    return imgout

def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    gamma = 5.0
    c = np.power(L-1,1-gamma)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            s = c*np.power(r,gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def PiecewiseLinear(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    rmin, rmax, vi_tri_rmin, vi_tri_rmax = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L-1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r < r1:
                s = s1/r1*r
            elif r < r2:
                s = (s2-s1)/(r2-r1)*(r-r1) + s1
            else:
                s = (L-1-s2)/(L-1-r2)*(r-r2) + s2
            imgout[x,y] = np.uint8(s)
    return imgout

def Histogram(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,L), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)
    scale = 2000
    for r in range(0, L):
        cv2.line(imgout,(r,M-1),(r,M-1-int(scale*p[r])), (0,0,0))
    return imgout

def HistEqual(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            h[r] = h[r]+1
    p = h/(M*N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k+1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            imgout[x,y] = np.uint8((L-1)*s[r])
    return imgout

def HistEqualColor(imgin):
    B = imgin[:,:,0]
    G = imgin[:,:,1]
    R = imgin[:,:,2]
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    imgout = np.array([B, G, R])
    imgout = np.transpose(imgout, axes = [1,2,0]) 
    return imgout

def LocalHist(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout

def HistStat(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    mG, sigmaG = cv2.meanStdDev(imgin)
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[x+s,y+t]
            msxy, sigmasxy = cv2.meanStdDev(w)
            r = imgin[x,y]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*r)
            else:
                imgout[x,y] = r
    return imgout

def MyBoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 11
    n = 11
    w = np.ones((m,n))
    w = w/(m*n)

    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, M-b):
            r = 0.0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    r = r + w[s+a,t+b]*imgin[x+s,y+t]
            imgout[x,y] = np.uint8(r)
    return imgout

def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.ones((m,n))
    w = w/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout

def Threshold(imgin):
    temp = cv2.blur(imgin, (15,15))
    retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
    return imgout

def MedianFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 5
    n = 5
    w = np.zeros((m,n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(0, M):
        for y in range(0, N):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a,t+b] = imgin[(x+s)%M,(y+t)%N]
            w_1D = np.reshape(w, (m*n,))
            w_1D = np.sort(w_1D)
            imgout[x,y] = w_1D[m*n//2]
    return imgout

def Sharpen(imgin):
    # Đạo hàm cấp 2 của ảnh
    w = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    temp = cv2.filter2D(imgin,cv2.CV_32FC1,w)

    # Hàm cv2.Laplacian chỉ tính đạo hàm cấp 2
    # cho bộ lọc có số -4 chính giữa
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout
 
def Gradient(imgin):
    sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    # Đạo hàm cấp 1 theo hướng x
    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    # Đạo hàm cấp 1 theo hướng y
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    # Lưu ý: cv2.Sobel có hướng x nằm ngang
    # ngược lại với sách Digital Image Processing
    gx = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 1, dy = 0)
    gy = cv2.Sobel(imgin,cv2.CV_32FC1, dx = 0, dy = 1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L-1)
    imgout = imgout.astype(np.uint8)
    return imgout

# ==================================== Modified =====================================================
def negative_streamlit():
    # Làm âm ảnh (Negative)
    st.markdown('''<h2 class="subheader-text">
                3.1/ Làm âm ảnh (Negative)
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Chuyển đổi một bức ảnh sang dạng đảo ngược của nó, trong đó mối quan hệ giữa độ sáng và độ tối của các phần tử hình ảnh được đảo ngược. Điều này có nghĩa là những vùng sáng trở thành tối và ngược lại. Quá trình làm âm ảnh thường được thực hiện bằng cách đảo ngược giá trị của từng pixel trong ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_1, bt3_1_another,bt3_1_delete = st.columns(3)
    bt3_1_state = bt3_1.button("Xử lý demo",type="primary")
    bt3_1_another_state = bt3_1_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_1_state:
        cot1.image(path+"3_1.tif",use_column_width=True)
        img = cv2.imread(path+"3_1.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Negative(img),use_column_width=True)
        bt3_1_delete_state = bt3_1_delete.button("Xóa",type="primary")
    if bt3_1_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Negative(frame),use_column_width=True)
            bt3_1_delete_state = bt3_1_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
            
def logarit_streamlit():
    # Logarit ảnh
    st.markdown('''<h2 class="subheader-text">
                3.2/ Logarit ảnh
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Việc áp dụng logarit cho một ảnh là một phương pháp trong xử lý hình ảnh, thường được sử dụng để cải thiện độ tương phản của ảnh. Khi bạn áp dụng logarit cho mỗi giá trị pixel trong ảnh, đồng thời điều chỉnh một số tham số, bạn có thể tăng độ tương phản của ảnh một cách đáng kể. Phương pháp này có thể được sử dụng trong nhiều ngữ cảnh khác nhau, bao gồm chụp ảnh, xử lý hình ảnh y khoa, và các ứng dụng khác nơi độ tương phản của ảnh là một yếu tố quan trọng.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_2, bt3_2_another,bt3_2_delete = st.columns(3)
    bt3_2_state = bt3_2.button("Xử lý demo",type="primary")
    bt3_2_another_state = bt3_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_2_state:
        cot1.image(path+"3_2.tif",use_column_width=True)
        img = cv2.imread(path+"3_2.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Logarit(img),use_column_width=True)
        bt3_2_delete_state = bt3_2_delete.button("Xóa",type="primary")
    if bt3_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Logarit(frame),use_column_width=True)
            bt3_2_delete_state = bt3_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
            
def power_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.3/ Lũy thừa ảnh
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Lũy thừa ảnh là một phương pháp trong xử lý hình ảnh, thường được sử dụng để cải thiện độ tương phản của ảnh. Quá trình này áp dụng một hàm lũy thừa (power-law transformation) cho mỗi giá trị pixel trong ảnh. Phương pháp lũy thừa thường được sử dụng trong nhiều lĩnh vực xử lý ảnh, bao gồm chụp ảnh, y học, và các ứng dụng khác nơi điều chỉnh độ tương phản của ảnh là cần thiết.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_3, bt3_3_another,bt3_3_delete = st.columns(3)
    bt3_3_state = bt3_3.button("Xử lý demo",type="primary")
    bt3_3_another_state = bt3_3_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_3_state:
        cot1.image(path+"3_3.tif",use_column_width=True)
        img = cv2.imread(path+"3_3.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Power(img),use_column_width=True)
        bt3_3_delete_state = bt3_3_delete.button("Xóa",type="primary")
    if bt3_3_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Power(frame),use_column_width=True)
            bt3_3_delete_state = bt3_3_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def PiecewiseLinear_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.4/ Biến đổi tuyến tính từng phần
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Biến đổi tuyến tính từng phần (Piecewise Linear Transformation) là một phương pháp trong xử lý hình ảnh được sử dụng để thay đổi độ tương phản của ảnh thông qua các phép biến đổi tuyến tính được áp dụng cho từng phần của hình ảnh. ")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_4, bt3_4_another,bt3_4_delete = st.columns(3)
    bt3_4_state = bt3_4.button("Xử lý demo",type="primary")
    bt3_4_another_state = bt3_4_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_4_state:
        cot1.image(path+"3_4.jpg",use_column_width=True)
        img = cv2.imread(path+"3_4.jpg",cv2.IMREAD_GRAYSCALE)
        cot2.image(PiecewiseLinear(img),use_column_width=True)
        bt3_4_delete_state = bt3_4_delete.button("Xóa",type="primary")
    if bt3_4_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(PiecewiseLinear(frame),use_column_width=True)
            bt3_4_delete_state = bt3_4_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def Histogram_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.5/ Histogram
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để cân bằng histogram, chuyển đổi mức xám hoặc phân ngưỡng.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_5, bt3_5_another,bt3_5_delete = st.columns(3)
    bt3_5_state = bt3_5.button("Xử lý demo",type="primary")
    bt3_5_another_state = bt3_5_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_5_state:
        cot1.image(path+"3_5.tif",use_column_width=True)
        img = cv2.imread(path+"3_5.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Histogram(img),use_column_width=True)
        bt3_5_delete_state = bt3_5_delete.button("Xóa",type="primary")
    if bt3_5_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Histogram(frame),use_column_width=True)
            bt3_5_delete_state = bt3_5_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def HistogramEqualization_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.6/ Cân bằng Histogram
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Cân bằng histogram của ảnh là một phương pháp trong xử lý hình ảnh được sử dụng để làm cho phân phối độ sáng của ảnh trở nên đồng đều hơn trên toàn bức ảnh. Mục tiêu của việc cân bằng histogram là tối ưu hóa phân phối của các cấp độ độ sáng trong ảnh, từ đó cải thiện độ tương phản và chi tiết hình ảnh. Dùng để làm phẳng phân bố của các mức xám trong ảnh, từ đó làm tăng độ tương phản và làm cho ảnh trở nên rõ ràng hơn.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_6, bt3_6_another,bt3_6_delete = st.columns(3)
    bt3_6_state = bt3_6.button("Xử lý demo",type="primary")
    bt3_6_another_state = bt3_6_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_6_state:
        cot1.image(path+"3_6.jpg",use_column_width=True)
        img = cv2.imread(path+"3_6.jpg",cv2.IMREAD_GRAYSCALE)
        cot2.image(HistEqual(img),use_column_width=True)
        bt3_6_delete_state = bt3_6_delete.button("Xóa",type="primary")
    if bt3_6_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(HistEqual(frame),use_column_width=True)
            bt3_6_delete_state = bt3_6_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def HistEqualColor_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.7/ Cân bằng Histogram của ảnh màu
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Cân bằng histogram của ảnh màu thường được thực hiện bằng cách cân bằng histogram cho từng kênh màu (RGB hoặc các kênh màu khác tùy thuộc vào không gian màu sử dụng). Quá trình này giống với cân bằng histogram của ảnh xám, nhưng được áp dụng độc lập cho từng kênh màu. Quá trình này giúp làm cho phân bố màu sắc trở nên đồng đều, từ đó cải thiện độ tương phản và chi tiết trong ảnh màu.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_7, bt3_7_another,bt3_7_delete = st.columns(3)
    bt3_7_state = bt3_7.button("Xử lý demo",type="primary")
    bt3_7_another_state = bt3_7_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_7_state:
        cot1.image(path+"3_7.tif",use_column_width=True)
        img = cv2.imread(path+"3_7.tif",cv2.IMREAD_COLOR)
        cot2.image(HistEqualColor(img),use_column_width=True)
        bt3_7_delete_state = bt3_7_delete.button("Xóa",type="primary")
    if bt3_7_another_state:
        if uploaded_file is not None:
            image_np = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            cot1.image(frame,use_column_width=True)
            cot2.image(HistEqualColor(frame),use_column_width=True)
            bt3_7_delete_state = bt3_7_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def LocalHistogram_streamlit():
    # Local Histogram
    st.markdown('''<h2 class="subheader-text">
                3.8/ Local Histogram
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Là một phương pháp cải thiện độ tương phản và chi tiết của ảnh một cách hiệu quả, đặc biệt là trong các khu vực cụ thể của ảnh. Thay vì cân bằng histogram trên toàn bức ảnh, LHE tập trung vào việc cân bằng histogram trong các vùng nhỏ, cụ thể là các cửa sổ hoặc ô vuông nhỏ trên bức ảnh. Làm thay đổi độ tương phản và độ sáng trong ảnh dựa trên phân bố các mức xám trong các vùng nhỏ cục bộ của ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_8, bt3_8_another,bt3_8_delete = st.columns(3)
    bt3_8_state = bt3_8.button("Xử lý demo",type="primary")
    bt3_8_another_state = bt3_8_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_8_state:
        cot1.image(path+"3_8.tif",use_column_width=True)
        img = cv2.imread(path+"3_8.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(LocalHist(img),use_column_width=True)
        bt3_8_delete_state = bt3_8_delete.button("Xóa",type="primary")
    if bt3_8_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(LocalHist(frame),use_column_width=True)
            bt3_8_delete_state = bt3_8_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def StatisticHistogram_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.9/ Thống kê histogram
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Histogram là một biểu đồ thống kê được sử dụng để mô tả phân phối của một tập dữ liệu. Trong xử lý hình ảnh, histogram thường được sử dụng để biểu diễn phân phối độ sáng của các pixel trong ảnh. Dùng để cải thiện độ tương phản và độ sáng của tổng thể ảnh.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_9, bt3_9_another,bt3_9_delete = st.columns(3)
    bt3_9_state = bt3_9.button("Xử lý demo",type="primary")
    bt3_9_another_state = bt3_9_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_9_state:
        cot1.image(path+"3_9.tif",use_column_width=True)
        img = cv2.imread(path+"3_9.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(HistStat(img),use_column_width=True)
        bt3_9_delete_state = bt3_9_delete.button("Xóa",type="primary")
    if bt3_9_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(HistStat(frame),use_column_width=True)
            bt3_9_delete_state = bt3_9_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def blur_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.10/ Lọc box
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để giảm nhiễu và làm mờ các chi tiết không mong muốn trong ảnh. Phương pháp này nhằm làm giảm sự biến động cục bộ trong ảnh để tạo ra một phiên bản làm mờ của ảnh gốc.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_10, bt3_10_another,bt3_10_delete = st.columns(3)
    bt3_10_state = bt3_10.button("Xử lý demo",type="primary")
    bt3_10_another_state = bt3_10_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_10_state:
        cot1.image(path+"3_10.tif",use_column_width=True)
        img = cv2.imread(path+"3_10.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(cv2.blur(img,(21,21)),use_column_width=True)
        bt3_10_delete_state = bt3_10_delete.button("Xóa",type="primary")
    if bt3_10_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(cv2.blur(frame,(21,21)),use_column_width=True)
            bt3_10_delete_state = bt3_10_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def GaussianBlur_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.11/ Lọc Gauss
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để làm mờ và giảm nhiễu.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_11, bt3_11_another,bt3_11_delete = st.columns(3)
    bt3_11_state = bt3_11.button("Xử lý demo",type="primary")
    bt3_11_another_state = bt3_11_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_11_state:
        cot1.image(path+"3_11.tif",use_column_width=True)
        img = cv2.imread(path+"3_11.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(cv2.GaussianBlur(img,(43,43),7.0),use_column_width=True)
        bt3_11_delete_state = bt3_11_delete.button("Xóa",type="primary")
    if bt3_11_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(cv2.GaussianBlur(frame,(43,43),7.0),use_column_width=True)
            bt3_11_delete_state = bt3_11_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def Threshold_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.12/ Phân ngưỡng
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Trong xử lý hình ảnh, phân ngưỡng (thresholding) là một phương pháp được sử dụng để chia ảnh thành các vùng có giá trị pixel khác nhau dựa trên một giá trị ngưỡng (threshold) nhất định. Mục tiêu của phân ngưỡng là tạo ra một ảnh nhị phân, trong đó mỗi pixel được đặt vào một trong hai nhóm: một nhóm được coi là 'đen' (thường là giá trị 0) hoặc 'trắng' (thường là giá trị cực đại của pixel, ví dụ 255 trong ảnh 8-bit).")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_12, bt3_12_another,bt3_12_delete = st.columns(3)
    bt3_12_state = bt3_12.button("Xử lý demo",type="primary")
    bt3_12_another_state = bt3_12_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_12_state:
        cot1.image(path+"3_12.tif",use_column_width=True)
        img = cv2.imread(path+"3_12.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Threshold(img),use_column_width=True)
        bt3_12_delete_state = bt3_12_delete.button("Xóa",type="primary")
    if bt3_12_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Threshold(frame),use_column_width=True)
            bt3_12_delete_state = bt3_12_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
    
def MedianFilter_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.13.1/ Lọc median
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để làm mờ và giảm nhiễu.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_13_1, bt3_13_1_another,bt3_13_1_delete = st.columns(3)
    bt3_13_1_state = bt3_13_1.button("Xử lý demo",type="primary")
    bt3_13_1_another_state = bt3_13_1_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_13_1_state:
        cot1.image(path+"3_13_1.tif",use_column_width=True)
        img = cv2.imread(path+"3_13_1.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(MedianFilter(img),use_column_width=True)
        bt3_13_1_delete_state = bt3_13_1_delete.button("Xóa",type="primary")
    if bt3_13_1_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(MedianFilter(frame),use_column_width=True)
            bt3_13_1_delete_state = bt3_13_1_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def Sharpen_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.13.2/ Sharpen
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để làm nổi bật các đặc trưng và tăng độ rõ nét của ảnh. Thường được dùng làm nổi bật các biên cạnh và chi tiết.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_13_2, bt3_13_2_another,bt3_13_2_delete = st.columns(3)
    bt3_13_2_state = bt3_13_2.button("Xử lý demo",type="primary")
    bt3_13_2_another_state = bt3_13_2_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_13_2_state:
        cot1.image(path+"3_13_2.tif",use_column_width=True)
        img = cv2.imread(path+"3_13_2.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Sharpen(img),use_column_width=True)
        bt3_13_2_delete_state = bt3_13_2_delete.button("Xóa",type="primary")
    if bt3_13_2_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Sharpen(frame),use_column_width=True)
            bt3_13_2_delete_state = bt3_13_2_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")

def Gradient_streamlit():
    st.markdown('''<h2 class="subheader-text">
                3.13.3/ Gradient
                </h2>
                ''',unsafe_allow_html=True)
    st.write("- Dùng để làm nổi bật các biên cạnh và các chi tiết trong ảnh. Phương pháp này dựa trên việc tính toán độ dốc (gradient) của ảnh, tức là sự thay đổi độ sáng giữa các điểm ảnh liền kề.")
    # Upload file
    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, PNG, TIF file",
        type=["jpg", "jpeg", "png", "tif"],
        help="Scanned file are not supported yet!",
    )
    bt3_13_3, bt3_13_3_another,bt3_13_3_delete = st.columns(3)
    bt3_13_3_state = bt3_13_3.button("Xử lý demo",type="primary")
    bt3_13_3_another_state = bt3_13_3_another.button("Xử lý ảnh khác",type="primary")
    cot1, cot2 = st.columns(2)
    if bt3_13_3_state:
        cot1.image(path+"3_13_3.tif",use_column_width=True)
        img = cv2.imread(path+"3_13_3.tif",cv2.IMREAD_GRAYSCALE)
        cot2.image(Gradient(img),use_column_width=True)
        bt3_13_3_delete_state = bt3_13_3_delete.button("Xóa",type="primary")
    if bt3_13_3_another_state:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            cot1.image(frame,use_column_width=True)
            cot2.image(Gradient(frame),use_column_width=True)
            bt3_13_3_delete_state = bt3_13_3_delete.button("Xóa",type="primary")
        else:
            st.warning("Vui lòng upload file!")
            
    st.markdown('''<h2 class="title">
                Hết.
                </h2>
                ''',unsafe_allow_html=True)



# File ảnh
path = ".\\Module\\XuLyAnh\\img\\chuong3\\"

def Chuong3_streamlit():
    st.markdown('''<h2 class="title">
                Chương 3: Biến đổi độ sáng và lọc trong không gian
                </h2>
                ''',unsafe_allow_html=True)
    
    
    tab_names = ["Làm âm ảnh (Negative)", "Logarit ảnh","Lũy thừa ảnh","Biến đổi tuyến tính từng phần",
                 "Histogram","Cân bằng Histogram","Cân bằng Histogram của ảnh màu",
                 "Local Histogram","Thống kê histogram","Lọc box","Lọc Gauss",
                 "Phân ngưỡng","Lọc median","Sharpen","Gradient"]
    selected_tab = st.sidebar.radio("Chọn nội dung xử lý:", tab_names)
    
    if selected_tab=="Làm âm ảnh (Negative)":
        negative_streamlit()
    if selected_tab=="Logarit ảnh":
        logarit_streamlit()
    if selected_tab=="Lũy thừa ảnh":
        power_streamlit()
    if selected_tab=="Biến đổi tuyến tính từng phần":
        PiecewiseLinear_streamlit()
    if selected_tab=="Histogram":
        Histogram_streamlit()
    if selected_tab=="Cân bằng Histogram":
        HistogramEqualization_streamlit()
    if selected_tab=="Cân bằng Histogram của ảnh màu":
        HistEqualColor_streamlit()
    if selected_tab=="Local Histogram":
        LocalHistogram_streamlit()
    if selected_tab=="Thống kê histogram":
        StatisticHistogram_streamlit()
    if selected_tab=="Lọc box":
        blur_streamlit()
    if selected_tab=="Lọc Gauss":
        GaussianBlur_streamlit()
    if selected_tab=="Phân ngưỡng":
        Threshold_streamlit()
    if selected_tab=="Lọc median":
        MedianFilter_streamlit()
    if selected_tab=="Sharpen":
        Sharpen_streamlit()
    if selected_tab=="Gradient":
        Gradient_streamlit()
    
    
















