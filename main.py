# ========================================== import library ============================================
import streamlit as st
from streamlit_option_menu import option_menu








# ======================================= Cấu hình streamlit ===========================================
# Cấu hình trang Streamlit
st.set_page_config(page_title='Đồ án môn xử lý ảnh', page_icon=':cyclone:', layout='wide')

# Cảnh báo outlier
st.warning("""
    Trang web này phục vụ cho mục đích báo cáo đồ án môn xử lý ảnh 
    được thực hiện bởi Võ Hoài Trọng - 21133112! 
    Vui lòng liên hệ email trongvo250403@gmail.com nếu có vấn đề về trang web!
""")

# Nhúng css
with open(".\\GUI\\css\\streamlit.css") as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Sidebar 
with st.sidebar:
    selected = option_menu("Hãy lựa chọn chức năng mà bạn muốn",["Giới thiệu","Giải phương trình bậc 2","Nhận diện khuôn mặt (OpenCV)","Nhận dạng đối tượng yolov4 (OpenCV)","Nhận dạng chữ viết tay MNIST","Nhận dạng 5 loại trái cây","Xử lý ảnh","Phần làm thêm"],
                           icons=['','key','people','','book','apple','camera','plus'], menu_icon="cast", default_index=0,
                           styles={
                                "container": {"font-family": "Monospace"},
                                "icon": {"color":"#71738d"}, 
                                "nav-link": {"--hover-color": "#d2cdfa","font-family": "Monospace"},
                                "nav-link-selected": {"font-family": "Monospace","background-color": "#a9a9ff"},
                            }
                           )
    
# ===================================== Page Home =================================
def home():
    st.markdown(
    """
    <style>
        .stHeadingContainer span {
            color: #000000;
            font-size: 36px;
            text-align: center;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 20px;
        }

        [data-testid="stMarkdownContainer"] {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True
    )
    
    # Title
    st.markdown('<h1 class="title">Báo cáo đồ án môn xử lý ảnh</h1>', unsafe_allow_html=True)

    # Logo trường
    st.image("GUI/img/HCMUTE-fit.png")
    
    # information
    st.divider()
    st.markdown('<h1 class="title">Sinh viên thực hiện</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.text("")
        st.markdown(
            """
            ## Võ Hoài Trọng
            ##### MSSV: 21133112
            ##### Mã lớp: DIPR430685_23_1_01 (Thứ tư)
            ##### Giảng viên: ThS. Trần Tiến Đức
            """
        )
    with col2:
        st.image(".\\GUI\\img\\information.png")

    st.divider()
    st.markdown('<h1 class="title">Nội dung project</h1>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("6 chức năng chính")
        st.write("📖Giải phương trình bậc 2")
        st.write("📖Nhận diện khuôn mặt (opencv)")
        st.write("📖Nhận dạng đối tượng yolo4 (opencv)")
        st.write("📖Nhận dạng chữ viết tay MNIST")
        st.write("📖Nhận dạng 5 loại trái cây")
        st.write("📖Xử lý ảnh")
    with col2:
        st.subheader("Phần làm thêm")
        st.write("📖Nhận dạng gương mặt, giới tính và độ tuổi")
        st.write("📖Nhận diện biển số xe Yolo v8 (Custom)")
        st.write("📖Nhận dạng mũ bảo hiểm Yolo v8 (Custom)")
        st.write("📖Nhận diện bài tây (BlackJack)")
        st.write("📖Nhận diện chữ viết tay nâng cao")
        st.write("📖Nhận dạng gian lận trong thi cử")

if selected == "Giới thiệu":
    home()
    
# ================================ 1. Giải phương trình bậc 2 ========================
from Module.PhuongTrinhBac2.giai_pt_bac_2 import ptbac2_streamlitshow
def PT_Bac_2():
    # Title
    st.markdown('<h1 class="title">Giải phương trình bậc 2</h1>', unsafe_allow_html=True)
    st.image("GUI/img/ptbac2.png")
    st.write("- Trên trang web của chúng tôi, chúng tôi cung cấp một công cụ giải phương trình bậc 2 đơn giản và hiệu quả. Người dùng chỉ cần nhập các hệ số của phương trình và nhận kết quả ngay lập tức. Giao diện dễ sử dụng và tính năng kiểm tra đầu vào giúp đảm bảo tính chính xác của kết quả. Đây là một công cụ hữu ích để giải quyết nhanh chóng các phương trình bậc 2 trực tuyến.")
    ptbac2_streamlitshow()
    
if selected == "Giải phương trình bậc 2":
    PT_Bac_2()

# ================================ 2. Page Face Recognize=============================
from Module.FaceRecognize.mainFace import mainface
def face_recognize():
    mainface()

if selected == "Nhận diện khuôn mặt (OpenCV)":
    face_recognize()


# ================================= 3. Page Object Detect ============================
from Module.ObjectDetection.streamlit_yolov4_detect import streamlit_yolov4

def object_detect():
    # Title
    st.markdown('<h1 class="title">Nhận dạng đối tượng yolov4 (OpenCV)</h1>', unsafe_allow_html=True)

    # Logo fruit
    st.image("GUI/img/object_detect.jpg")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model của Yolov4 để nhận diện các đối tượng có trong danh sách các đối tượng đã được train của Yolov4 
                <h2 class="subheader-text">=> Hãy upload ảnh hoặc video bạn muốn chúng tôi nhận dạng ở bên dưới nhé! Ngoài ra bạn cũng có thể mở webcam để nhận dạng (Khuyến cáo máy mạnh)!</h2>
                <h2 class="subheader-text warning">Lưu ý: Hãy click button xóa để dọn sạch vùng nhớ tạm thời và kết quả khi không cần sử dụng nữa nhé!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
    streamlit_yolov4()
    
if selected == "Nhận dạng đối tượng yolov4 (OpenCV)":
    object_detect()

# ============================== 4. Page HandWriting Detect ==========================
from Module.HandWriting.MNIST.MNIST_streamlit import MNIST_streamlit_show
from Module.HandWriting.Handwriting.handwriting_streamlit import handwriting_streamlit_show
def handwriting_detect():
    # Title
    st.markdown('<h1 class="title">Nhận dạng chữ viết tay MNIST</h1>', unsafe_allow_html=True)

    # Logo fruit
    st.image("GUI/img/handwriting_detect.jpg")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ nhận diện chữ viết tay MNIST! Hãy click tạo ảnh sau đó chúng tôi sẽ nhận diện các chữ số có trong ảnh đó!
                </h2>
                ''',unsafe_allow_html=True)
    
    MNIST_streamlit_show()
    
    # Title
    st.markdown('<h1 class="title">Phần làm thêm</h1>', unsafe_allow_html=True)
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model đã được train sẵn để nhận diện chữ viết tay 
                <h2 class="subheader-text">(nguồn Github: https://github.com/githubharald/HTRPipeline)</h2>
                <h2 class="subheader-text">=> Hãy upload ảnh bạn muốn chúng tôi nhận dạng ở bên dưới nhé!</h2>
                <h2 class="subheader-text warning">Lưu ý: Hãy click button xóa để dọn sạch vùng nhớ tạm thời và kết quả khi không cần sử dụng nữa nhé!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
    handwriting_streamlit_show()
    
            
if selected == "Nhận dạng chữ viết tay MNIST":
    handwriting_detect()
    
# ================================= 5. Page Fruit Detect =============================
from Module.FruitDetection.fruitdetect_streamlit import fruitdetect_streamlit_show
def fruit_detect():
    # Title
    st.markdown('<h1 class="title">Nhận dạng 5 loại trái cây</h1>', unsafe_allow_html=True)

    # Logo fruit
    st.image("GUI/img/fruit_banner.png")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ sử dụng một model đã được train để nhận diện 5 loại trái cây sau
                <h2 class="subheader-text">1. Orange    2. Tomato   3. Carrot   4. Bell pepper  5. Grape</h2>
                <h2 class="subheader-text">=> Hãy upload ảnh hoặc video bạn muốn chúng tôi nhận dạng ở bên dưới nhé!</h2>
                <h2 class="subheader-text warning">Lưu ý: Hãy click button xóa để dọn sạch vùng nhớ tạm thời và kết quả khi không cần sử dụng nữa nhé!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
    fruitdetect_streamlit_show()
               
if selected == "Nhận dạng 5 loại trái cây":
    fruit_detect()
    
    
# ==================================== 6. Xử lý ảnh ===================================
from Module.XuLyAnh.Chapter03 import Chuong3_streamlit
from Module.XuLyAnh.Chapter04 import Chuong4_streamlit
from Module.XuLyAnh.Chapter05 import Chuong5_streamlit
from Module.XuLyAnh.Chapter09 import Chuong9_streamlit
if selected == "Xử lý ảnh":
    # Title
    st.markdown('<h1 class="title">Xử Lý Ảnh</h1>', unsafe_allow_html=True)

    # Logo
    st.image("GUI/img/xulyanh.jpg")
    
    st.markdown('''<h2 class="subheader-text">
                - Ở chức năng này, chúng ta sẽ áp dụng các kiến thức học được từ các chương và tiến hành xử lý các ảnh mẫu
                <h2 class="subheader-text warning">Hãy chọn các chương để thử ngay!</h2>
                </h2>
                ''',unsafe_allow_html=True)
    
    sub_items = ["Chương 3","Chương 4" ,"Chương 5","Chương 9"]
    selected_sub_item = st.sidebar.selectbox("Chọn chương xử lý ảnh", sub_items)
    if selected_sub_item == "Chương 3":
        Chuong3_streamlit()
    if selected_sub_item == "Chương 4":
        Chuong4_streamlit()
    if selected_sub_item == "Chương 5":
        Chuong5_streamlit()
    if selected_sub_item == "Chương 9":
        Chuong9_streamlit()
        
        
        
# ================================= 7. Phần làm thêm ==================================
from Module.LicensePlateRecognize.LicensePlate_Recognized_streamlit import LicensePlate_Recognized_streamlit_show
from Module.BlackJackRecognize.BlackJack_streamlit import BlackJack_streamlit_show
from Module.HelmetDetection.HelmetDetect_streamlit import HelmetDectected_streamlit_show
from Module.FaceAgeGenderDectected.FaceAgeGenderDectected_streamlit import FaceAgeGenderDectected_streamlit_show
from Module.ExamCheatingDetection.ExamCheatingDetect_streamlit import ExamCheatingDetect_streamlit_show

if selected == "Phần làm thêm":
    sub_items = ["Nhận dạng gương mặt, giới tính và độ tuổi", "Nhận diện biển số xe Yolo v8 (Custom)", 
                 "Nhận dạng mũ bảo hiểm Yolo v8 (Custom)", "Nhận diện bài tây (BlackJack)",
                 "Nhận dạng gian lận trong thi cử (Cheating Exam)"]
    selected_sub_item = st.sidebar.selectbox("Chọn phần làm thêm", sub_items)
    if selected_sub_item == "Nhận dạng gương mặt, giới tính và độ tuổi":
        FaceAgeGenderDectected_streamlit_show()
    if selected_sub_item == "Nhận diện biển số xe Yolo v8 (Custom)":
        LicensePlate_Recognized_streamlit_show()
    if selected_sub_item == "Nhận dạng mũ bảo hiểm Yolo v8 (Custom)":
        HelmetDectected_streamlit_show()
    if selected_sub_item == "Nhận diện bài tây (BlackJack)":
        BlackJack_streamlit_show()
    if selected_sub_item == "Nhận dạng gian lận trong thi cử (Cheating Exam)":
        ExamCheatingDetect_streamlit_show()