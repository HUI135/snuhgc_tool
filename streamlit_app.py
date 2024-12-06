import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import requests
import plotly.figure_factory as ff
import plotly.graph_objects as go
import time
from scipy.stats import chi2_contingency, ttest_ind, levene, kruskal, shapiro, f_oneway
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import hmac
import hashlib
import base64
import datetime
import boto3
from botocore.client import Config
# from pgmpy.estimators import HillClimbSearch, BicScore
# import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt
import networkx as nx
import openpyxl
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os
import tempfile
import traceback
# from causallearn.utils.GraphUtils import graph_to_adjacency_matrix

# wide format
st.set_page_config(layout="wide")

############################
######### Homepage #########
############################

PASSWORD = "snuhgchc"  # Change this to your desired password

# 이미지와 제목 표시 함수
def display_header():
    image_url = 'http://www.snuh.org/upload/about/hi/15e707df55274846b596e0d9095d2b0e.png'
    title_html = "<h1 style='display: inline-block; margin: 0;'>🏥 GC DataRoom</h1>"
    contact_info_html = """
    <div style='text-align: left; font-size: 20px; color: grey;'>
    오류 문의: 헬스케어연구소 데이터 연구원 김희연 (hui135@snu.ac.kr)</div>
    """

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(image_url, width=200)
    with col2:
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(contact_info_html, unsafe_allow_html=True)
    st.divider()

# Initialize session state keys
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # Default: not logged in
if "loading_complete" not in st.session_state:
    st.session_state.loading_complete = False  # Loading state
if "header_displayed" not in st.session_state:
    st.session_state.header_displayed = False  # Track header display status

# 로그인 함수
def login():
    """Handles user login."""
    if not st.session_state.logged_in:  # Show login form only if not logged in
        st.sidebar.title("로그인")
        password = st.sidebar.text_input("비밀번호를 입력하세요", type="password")
        if password == PASSWORD:
            st.sidebar.success("비밀번호가 일치합니다.")
            st.session_state.logged_in = True
            st.session_state.loading_complete = False  # Reset loading state
            st.session_state.header_displayed = False  # Reset header status
            return True
        elif password:
            st.sidebar.error("비밀번호가 일치하지 않습니다.")
        return False
    return True

# Persistent Header
def display_persistent_header():
    """Display header once and persist across interactions."""
    display_header()  # Always call header display function
    st.session_state.header_displayed = True

# Main App Logic
if login():  # If logged in, show the rest of the app
    # 로딩 상태에서 Header를 표시
    if not st.session_state.loading_complete:
        display_persistent_header()  # Header 표시 during loading
        with st.spinner("Loading..."):
            time.sleep(3)  # Simulate loading time
        st.session_state.loading_complete = True  # Mark loading as complete
        st.session_state.header_rendered = True  # Track header rendering during loading

    # 로그인 후 상태에서 Header를 유지
    if not st.session_state.header_rendered:  # Prevent duplicate Header rendering
        display_persistent_header()
        st.session_state.header_rendered = True  # Mark Header as rendered

    # Sidebar with functionality options after login
    st.sidebar.empty()  # Clear the sidebar
    st.sidebar.title("기능 선택")
    page = st.sidebar.selectbox(
        "✔️ 사용하실 기능을 선택해주세요:",
        ["-- 선택 --", "🔔 사용설명서", "♻️ 인과관계 추론", "📝 피봇 변환", "📝 데이터 코딩", "📝 판독문 코딩", "📊 시각화", "📊 특성표 생성", "💻 로지스틱 회귀분석", "💻 생존분석", "⛔ 오류가 발생했어요"],
        index=0  # Default to "-- 선택 --"
    )

    # Page-specific content
    if page == "-- 선택 --":
        # Checkbox for updates
        toggle = st.checkbox("**📅 24.12.11 📅 Update 사항 자세히보기**")

        if toggle:
            # Toggle 활성화 시 Markdown 출력
            st.markdown("""
            ### 주요 업데이트 사항
            - (예시) **📝 피봇 변환** : 새로운 기능 추가
            - (예시) **📊 시각화** : 기능 추가 - 파이 차트 생성

            **세부 오류 수정**
            - (예시) 오류 수정 : `There are multiple radio elements with the same auto-generated ID`
            - (예시) 파일을 읽는 중 오류가 발생했습니다 : `There are multiple radio elements with the same auto-generated ID`
            - (예시) 시각화 기능 문제 : `TypeError: pie() got an unexpected keyword argument 'x'`
            """)
        else:
            # Toggle 비활성화 시 Info 출력
            st.info(" **환영합니다!** 좌측 사이드바에서 원하시는 기능을 선택해주세요.", icon='💡')
            
    elif page == "🔔 사용설명서":
        st.session_state.header_displayed = False
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">🔔  사용설명서</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;기능 사용설명법을 영상을 통해 살펴보세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        st.markdown("<h4 style='color:grey;'>어떤 기능이 궁금하신가요?</h4>", unsafe_allow_html=True)
        selected = st.selectbox("✔️ 사용설명서를 보실 기능을 선택해주세요:", options=["-- 선택 --", "♻️ 인과관계 추론", "📝 피봇 변환", "📝 데이터 코딩", "📝 판독문 코딩", "📊 시각화", "📊 특성표 생성", "💻 로지스틱 회귀분석", "💻 생존분석"])
        if selected == "-- 선택 --":
            st.write()
        elif selected == "📝 판독문 코딩":
            st.video("https://youtu.be/uE45G40TnTE")

    elif page == "📝 피봇 변환":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📝 피봇 변환</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;환자의 여러 내원 결과가 포함된 데이터를 열 기반으로 정리하는 기능입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # Track the uploaded file in session state to reset the UI when a new file is uploaded
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 환자의 여러 내원 데이터가 행으로 축적되어있는 데이터 파일을 업로드해주세요:", type=["csv", "xlsx"])


        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.session_state.df = df
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="📝 피봇 변환")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    st.divider()

                st.markdown("<h4 style='color:grey;'>데이터 정보입력</h4>", unsafe_allow_html=True)

                # 유저에게 피벗할 기준 열 선택 (selectbox에 '-- 선택 --' 추가)
                id_column = st.selectbox("✔️ 환자를 구분할 ID 혹은 RID 열을 선택해주세요:", ["-- 선택 --"] + list(df.columns))
                if id_column == "-- 선택 --":
                    st.write(" ")
                    st.stop()

                date_column = st.selectbox("✔️ 방문을 구분할 Date 열을 선택해주세요:", ["-- 선택 --"] + list(df.columns))
                if date_column == "-- 선택 --":
                    st.write(" ")
                    st.stop()

                # 날짜 변환 및 정렬
                try:
                    # 날짜 변환 및 유효성 검사
                    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')  # Ensure valid date conversion

                    # 변환된 날짜 열의 유효한 날짜 개수 확인
                    if df[date_column].isnull().all():
                        st.error("날짜 열에 결측값이 존재하여 변환을 수행할 수 없습니다.")
                        st.stop()  # 더 이상의 코드 실행을 중단
                    else:
                        # 유효하지 않은 날짜 행 삭제
                        df = df.dropna(subset=[date_column])  # Drop rows with invalid dates
                        df = df.sort_values(by=[id_column, date_column]).reset_index(drop=True)
                except Exception as e:
                    st.error("날짜 열을 처리하는 중 오류가 발생했습니다.")
                    st.stop()
                df = df.dropna(subset=[date_column])  # Drop rows with invalid dates
                df = df.sort_values(by=[id_column, date_column]).reset_index(drop=True)

                # 열 번호 붙이기
                df['row_number'] = df.groupby(id_column).cumcount() + 1
                df_pivot = df.pivot(index=id_column, columns='row_number')
                df_pivot.columns = [f"{col}_{num}" for col, num in df_pivot.columns]

                df_pivot.reset_index(inplace=True)

                # 결과 표시
                st.divider()
                st.header("📝 피봇 변환 결과", divider='rainbow')

                total_len = len(df)  # Total number of rows
                unique_len = df[id_column].nunique()  # Number of unique IDs (patients)

                # Counting the number of patients who visited once, twice, and three times
                visit_counts = df[id_column].value_counts()
                once_len = (visit_counts == 1).sum()
                twice_len = (visit_counts == 2).sum()
                third_len = (visit_counts == 3).sum()
                max_len = visit_counts.max()
                avg_len = round(visit_counts.mean(), 2)

                # Displaying the results using Streamlit markdown
                st.markdown(
                    f"<p style='font-size:16px; color:#0D47A1;'><strong>- 전체 데이터에는 총 '{total_len:,}'개의 행이 있으며, 이 중 '{unique_len:,}'명의 환자 데이터가 포함되어 있습니다.</strong></p>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<p style='font-size:16px; color:#0D47A1;'><strong>- 한 번 내원한 환자는 '{once_len:,}'명, 두 번 내원한 환자는 '{twice_len:,}'명, 세 번 내원한 환자는 '{third_len:,}'명입니다.</strong></p>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<p style='font-size:16px; color:#0D47A1;'><strong>- 환자의 최대 내원 횟수는 '{max_len}'이며, 평균 내원 횟수는 '{avg_len}'회입니다.</strong></p>",
                    unsafe_allow_html=True
                )

                # Simulate a long-running process
                def long_running_process():
                    # Replace this loop with your actual computation or loading process
                    for i in range(100):
                        time.sleep(0.1)  # Example processing time for each step

                with st.spinner("Loading... 30초 가량의 로딩이 소요됩니다."):
                    long_running_process()  # Run your process here instead of time.sleep()

                st.success("작업이 완료되었습니다!")
                st.write(" ")

                # 세션 상태에 저장
                st.session_state.df_pivot = df_pivot
                st.markdown("<h4 style='color:grey;'>피봇 데이터</h4>", unsafe_allow_html=True)
                st.dataframe(df_pivot)

                st.write(" ")
                st.markdown("<h4 style='color:grey;'>피봇 데이터 다운로드</h4>", unsafe_allow_html=True)

                # Display the file format selection radio button for original data download
                export_format_original = st.radio("✔️ 파일 형식을 선택해주세요:", options=["CSV", "Excel"], key="export_format_original")

                # Handle original data download
                if export_format_original:
                    if export_format_original == "CSV":
                        csv = st.session_state.df_pivot.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="CSV 다운로드",
                            data=csv,
                            file_name="pivot_data.csv",
                            mime='text/csv'
                        )
                    elif export_format_original == "Excel":
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            st.session_state.df_pivot.to_excel(writer, index=False)
                        buffer.seek(0)
                        st.download_button(
                            label="Excel 다운로드 (원본 자료)",
                            data=buffer,
                            file_name="pivot_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                # Initialize session state for df_pivot and button states if they are not already present
                if "df_pivot" not in st.session_state:
                    st.session_state.df_pivot = df_pivot  # Load df_pivot data if not already loaded
                if "filter_button_pressed" not in st.session_state:
                    st.session_state.filter_button_pressed = False
                if "plot_button_pressed" not in st.session_state:
                    st.session_state.plot_button_pressed = False
                if "download_button_pressed" not in st.session_state:
                    st.session_state.download_button_pressed = False
                if "download_filtered_button_pressed" not in st.session_state:
                    st.session_state.download_filtered_button_pressed = False

                # 피벗에 사용된 열 제외
                excluded_columns = [id_column, date_column]

                # 시각화
                if st.session_state.get("df_pivot") is not None:
                    st.divider()
                    st.header("📝 피봇 데이터 시각화", divider="rainbow")

                    # 원본 데이터프레임에서 제외된 열을 제거하여 선택 가능한 열 생성
                    original_columns = [col for col in st.session_state.df.columns if col not in excluded_columns]
                    selected_column_base = st.selectbox("✔️ 시각화할 변수 열을 선택해주세요:", ["-- 선택 --"] + original_columns)

                    if selected_column_base != "-- 선택 --":
                        # 피벗 데이터프레임에서 선택한 열에 해당하는 관련 열 필터링
                        visit_columns = [
                            col for col in st.session_state.df_pivot.columns 
                            if col.startswith(selected_column_base + '_')
                        ]

                        if visit_columns:
                            # 각 방문 회차별 평균 계산
                            mean_values = st.session_state.df_pivot[visit_columns].mean()

                            # Function to plot average LabResult changes across visits
                            def plot_average_changes(mean_values):
                                visit_numbers = [int(col.split('_')[1]) for col in mean_values.index]
                                avg_values = mean_values.values

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=visit_numbers,
                                    y=avg_values,
                                    mode='lines+markers',
                                    name="Average LabResult",
                                    marker=dict(symbol='circle', size=10)
                                ))

                                fig.update_xaxes(title_text="Visits", dtick=1)
                                fig.update_yaxes(title_text=f"Average {selected_column_base}")
                                fig.update_layout(
                                    height=600,
                                    width=900,
                                    title_text=f"Average {selected_column_base} Trends Across Visits",
                                    showlegend=True
                                )
                                return fig

                            # Generate and display the plot
                            fig = plot_average_changes(mean_values)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("선택한 변수와 관련된 열이 피봇 데이터프레임에 없습니다.")

                # Filter button in a row
                if st.session_state.get("df_pivot") is not None:
                    # Display a divider and a section header
                    st.divider()
                    st.header("📝 피봇 데이터 필터링", divider="rainbow")
                    num = st.selectbox("✔️ 최대 내원 횟수를 n회로 필터링합니다:", ["-- 선택 --"] + list(range(1, max_len)))

                    if num != "-- 선택 --":
                        # Determine columns to keep based on the selected max visit count
                        columns_to_keep = [col for col in st.session_state.df_pivot.columns
                                        if not any(col.endswith(f"_{i}") for i in range(num + 1, max_len + 1))]

                        # Filter the DataFrame based on selected columns
                        df_pivot_filtered = st.session_state.df_pivot[columns_to_keep]
                        st.session_state.df_pivot_filtered = df_pivot_filtered

                        # Display the filtered DataFrame
                        st.dataframe(df_pivot_filtered, use_container_width=True)

                        st.write(" ")
                        st.markdown("<h4 style='color:grey;'>피봇(필터) 데이터 다운로드</h4>", unsafe_allow_html=True)

                        export_format_filtered = st.radio("✔️ 파일 형식을 선택해주세요:", options=["CSV", "Excel"], key="export_format_filtered")

                        # Handle filtered data download
                        if export_format_filtered:
                            if export_format_filtered == "CSV":
                                csv = st.session_state.df_pivot_filtered.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="CSV 다운로드 (필터 자료)",
                                    data=csv,
                                    file_name="pivot_data_filtered.csv",
                                    mime='text/csv'
                                )
                            elif export_format_filtered == "Excel":
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                    st.session_state.df_pivot_filtered.to_excel(writer, index=False)
                                buffer.seek(0)
                                st.download_button(
                                    label="Excel 다운로드 (필터 자료)",
                                    data=buffer,
                                    file_name="pivot_data_filtered.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )

            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")
            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")

    elif page == "📝 데이터 코딩":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📝 데이터 코딩</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;코딩이 필요한 데이터를 업로드하신 후 원하시는 코딩을 수행하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # Track the uploaded file in session state to reset the UI when a new file is uploaded
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 코딩을 수행하실 데이터 파일을 업로드해주세요:", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                # 데이터 미리보기 표시
                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="📝 데이터 코딩")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    # 열 선택창
                    st.divider()
                    st.header("📝 데이터 코딩", divider='rainbow')
                    st.markdown(
                        """
                        <style>
                        .custom-callout {
                            background-color: #f9f9f9;
                            padding: 10px;
                            border-radius: 10px;
                            border: 1px solid #d3d3d3;
                            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        }
                        .custom-callout p {
                            margin: 0;
                            color: #000000;
                            font-size: 14px;
                            line-height: 1.4;
                            text-align: left;
                        }
                        </style>
                        <div class="custom-callout">
                            <p><strong>하단에 생성할 코딩 열의 이름을 입력 후, 조건을 입력하면 코딩이 이뤄집니다. 조건에 포함되지 않는 경우, 0으로 코딩됩니다.</strong></p>
                            <p>🔔 주의!) 간단한 코딩 기능만을 제공하므로, 그외의 코딩이 필요하신 경우 문의를 부탁드립니다.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(" ")
                    st.write(" ")

                    columns = df.columns.tolist()
                    columns.insert(0, "-- 선택 --")

                    # 선택된 열을 기반으로 작업
                    st.session_state.df = df  # Ensure df is stored initially
                    columns = df.columns.tolist()

                    # 초기 상태 설정
                    if "codes" not in st.session_state:
                        st.session_state.codes = []  # 코딩 코드 리스트
                    if "conditions" not in st.session_state:
                        st.session_state.conditions = {}  # 코드별 조건 딕셔너리
                    if "conditions_complete" not in st.session_state:
                        st.session_state.conditions_complete = {}  # 코드별 완료된 조건 설명

                    # UI 구성
                    st.markdown("<h4 style='color:grey;'>코드 추가</h4>", unsafe_allow_html=True)

                    # 새로운 열 이름 입력
                    new_column_name = st.text_input("▶️ 생성할 데이터 열의 이름을 입력 후 엔터를 눌러주세요:")
                    st.session_state.new_column_name = new_column_name

                    if new_column_name:
                        if new_column_name not in df.columns:
                            df[new_column_name] = np.nan  # 기본적으로 NaN으로 채움
                            st.markdown(f"코딩 결과가 저장될 열: **{new_column_name}**", unsafe_allow_html=True)


                    def add_condition_ui(code):
                        """조건 설정 UI 생성 함수"""
                        st.divider()
                        st.markdown(f"<h5 style='color:grey;'>✅ 코드 {code}에 대한 조건 설정</h5>", unsafe_allow_html=True)

                        # 조건 추가/삭제 버튼
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"➕ 조건 추가", key=f"add_condition_{code}"):
                                st.session_state.conditions[code].append(
                                    {"column": None, "operator": None, "value": None, "logic": f"AND 조건 {len(st.session_state.conditions[code])}"}
                                )
                        with col2:
                            if st.button(f"❌ 조건 삭제", key=f"remove_condition_{code}"):
                                if len(st.session_state.conditions[code]) > 1:  # 최소 1개의 조건은 유지
                                    st.session_state.conditions[code].pop()

                        # 조건 UI 생성
                        for idx, cond in enumerate(st.session_state.conditions[code], start=1):
                            st.markdown(f"✔️ 조건 {idx}")
                            columns = st.columns([2, 2, 2, 2] if idx > 1 else [2, 2, 2])
                            if idx > 1:
                                cond["logic"] = columns[0].selectbox(
                                    "- 논리",
                                    options=[f"AND 조건 {idx - 1}", f"OR 조건 {idx - 1}", f"NOT 조건 {idx - 1}"],
                                    key=f"logic_{code}_{idx}",
                                )
                            cond["column"] = columns[-3].selectbox(
                                "- 사용할 열",
                                options=["-- 선택 --"] + df.columns.tolist(),
                                key=f"col_{code}_{idx}",
                            )
                            cond["value"] = columns[-2].number_input(
                                "- 값",
                                step=1,
                                key=f"value_{code}_{idx}",
                            )
                            cond["operator"] = columns[-1].selectbox(
                                "- 연산",
                                options=["이상", "이하", "미만", "초과", "같음"],
                                key=f"operator_{code}_{idx}",
                            )

                    # UI 구성
                    new_code_name = st.text_input("▶️ 추가할 코드를 입력하세요:")
                    if st.button("코드 추가"):  # 확인 버튼 추가
                        if new_code_name:  # 입력된 코드가 있는지 확인
                            if new_code_name not in st.session_state.codes:
                                st.session_state.codes.append(new_code_name)
                                st.success(f"코드 {new_code_name}가 추가되었습니다!")
                            else:
                                st.warning(f"코드 {new_code_name}는 이미 추가되어 있습니다.")
                        else:
                            st.warning("코드 이름을 입력하세요.")

                    # 입력된 코드 목록 표시 및 삭제 기능
                    if st.session_state.codes:
                        st.write(" ")
                        st.markdown("<h5>현재 입력된 코드 목록:</h5>", unsafe_allow_html=True)
                        codes_to_keep = st.session_state.codes.copy()
                        for code_name in st.session_state.codes:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"<strong><span style='color:#526E48;'>✅ 코드: {code_name}</span>", unsafe_allow_html=True)
                            with col2:
                                if st.button(f"❌ 삭제", key=f"delete_{code_name}"):
                                    codes_to_keep.remove(code_name)  # 리스트에서 코드 제거
                        st.session_state.codes = codes_to_keep

                    # "코딩 시작" 버튼
                    if st.button("🚀 코딩 시작"):
                        # 삭제 후 남은 코드들로 조건 설정 시작
                        st.session_state.remaining_codes = st.session_state.codes.copy()
                        for code in st.session_state.remaining_codes:
                            if code not in st.session_state.conditions:
                                st.session_state.conditions[code] = [
                                    {"column": None, "operator": None, "value": None, "logic": None},  # 조건 1
                                    {"column": None, "operator": None, "value": None, "logic": "AND 조건 1"},  # 조건 2
                                ]

                    # 조건 설정 UI
                    if "remaining_codes" in st.session_state and st.session_state.remaining_codes:
                        for code in st.session_state.remaining_codes:
                            add_condition_ui(code)

                        # 미처리 항목 처리 옵션
                        st.divider()
                        st.markdown("<h4 style='color:grey;'>코딩되지 않은 그 외 항목 처리 방법</h4>", unsafe_allow_html=True)

                        fill_option = st.radio("✔️ 처리 방법을 선택해주세요:", ("전부 0으로", "전부 99로", "전부 공백으로"))

                        # 코딩 완료 버튼
                        if st.button("🚀 코딩 완료"):
                            # 데이터 프레임 복사
                            coded_df = df.copy()

                            # 기본값 설정
                            if fill_option == "전부 0으로":
                                default_fill = 0
                            elif fill_option == "전부 99로":
                                default_fill = 99
                            elif fill_option == "전부 공백으로":
                                default_fill = None

                            coded_df[new_column_name] = default_fill  # 기본값으로 채움

                            # 사용된 열 추적
                            used_columns = set()

                            for code in st.session_state.remaining_codes:
                                conditions = st.session_state.conditions[code]
                                condition_query = []

                                # 조건 변환
                                for cond in conditions:
                                    if cond["column"] and cond["column"] != "-- 선택 --":
                                        column = cond["column"]
                                        operator = cond["operator"]
                                        value = cond["value"]

                                        # 조건에 따른 Pandas 쿼리식 생성
                                        if operator == "이상":
                                            query = f"({column} >= {value})"
                                        elif operator == "이하":
                                            query = f"({column} <= {value})"
                                        elif operator == "미만":
                                            query = f"({column} < {value})"
                                        elif operator == "초과":
                                            query = f"({column} > {value})"
                                        elif operator == "같음":
                                            query = f"({column} == {value})"
                                        else:
                                            continue

                                        if "logic" in cond and cond["logic"] and "NOT" in cond["logic"]:
                                            query = f"not {query}"  # NOT 조건 적용

                                        condition_query.append(query)
                                        used_columns.add(column)  # 사용된 열 추가

                                # 모든 조건을 조합하여 적용
                                if condition_query:
                                    final_query = " & ".join(condition_query)
                                    coded_df.loc[coded_df.query(final_query).index, new_column_name] = code

                            # 결과 저장
                            st.session_state.coded_df = coded_df
                            st.session_state.preview_df = coded_df[list(used_columns) + [new_column_name]]

                            st.success("코딩이 완료되었습니다. 데이터를 살펴보세요.")

                        # 데이터 미리보기 및 다운로드
                        if "preview_df" in st.session_state:
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>코딩 결과</h4>", unsafe_allow_html=True)
                            st.dataframe(st.session_state.preview_df, use_container_width=True)

                            # 파일 형식 선택
                            if "export_format" not in st.session_state:
                                st.session_state.export_format = "CSV"

                            st.divider()
                            st.markdown("<h4 style='color:grey;'>코딩 데이터 다운로드</h4>", unsafe_allow_html=True)
                            st.write("🔔 업로드하신 데이터에 새로 추가된 코딩 열이 포함된 형태로 데이터를 다운로드하실 수 있습니다.")
                            
                            export_format = st.radio(
                                "✔️ 파일 형식을 선택해주세요:",
                                options=["CSV", "Excel"],
                                key="export_format",
                                index=["CSV", "Excel"].index(st.session_state.export_format)
                            )

                            # CSV 다운로드
                            if export_format == "CSV":
                                csv = st.session_state.coded_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    label="CSV 다운로드",
                                    data=csv,
                                    file_name=f"{new_column_name}_table.csv",
                                    mime="text/csv",
                                    key="csv_download_button"
                                )

                            # Excel 다운로드
                            elif export_format == "Excel":
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                                    st.session_state.coded_df.to_excel(writer, index=False)
                                buffer.seek(0)  # Reset buffer position
                                st.download_button(
                                    label="Excel 다운로드",
                                    data=buffer,
                                    file_name=f"{new_column_name}_table.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="excel_download_button"
                                )


            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")
            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")

    elif page == "📝 판독문 코딩":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📝 판독문 코딩</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;판독문 텍스트 데이터를 업로드하신 후 원하시는 코딩을 수행하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # Track the uploaded file in session state to reset the UI when a new file is uploaded
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 판독문 텍스트 열을 포함한 파일을 업로드해주세요:", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                # 데이터 미리보기 표시
                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="📝 판독문 코딩")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    # 판독문 열 선택창
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>판독문 텍스트 열 선택</h4>", unsafe_allow_html=True)
                    columns = df.columns.tolist()
                    columns.insert(0, "-- 선택 --")

                    selected_column = st.selectbox("✔️ 코딩할 판독문 텍스트 열을 선택해주세요:", options=columns)
                    if selected_column != "-- 선택 --":
                        # 'coding' 열 추가
                        if 'coding' not in df.columns:
                            df['coding'] = np.nan  # 기본적으로 nan으로 채움

                        # 선택된 열을 기반으로 작업
                        st.session_state.df = df  # Ensure df is stored initially


                    # Session state initialization for phrases (reset on new file upload)
                    if 'phrases_by_code' not in st.session_state:
                        st.session_state.phrases_by_code = {}  # Session state to hold phrases and codes

                    st.divider()
                    st.header("📝 판독문 코딩", divider='rainbow')
                    st.markdown(
                        """
                        <style>
                        .custom-callout {
                            background-color: #f9f9f9;
                            padding: 10px;
                            border-radius: 10px;
                            border: 1px solid #d3d3d3;
                            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        }
                        .custom-callout p {
                            margin: 0;
                            color: #000000;
                            font-size: 14px;
                            line-height: 1.4;
                            text-align: left;
                        }
                        </style>
                        <div class="custom-callout">
                            <p><strong>하단에 코드와 함께 텍스트를 입력 시, 해당 텍스트가 포함된 판독문 행은 함께 입력된 코드로 코딩이 이뤄집니다.</p>
                            <br>
                            <p>🔔 주의!) 먼저 입력한 코드 내용보다 뒤에 입력한 코드 내용에 높은 우선순위가 부여됩니다.</p>
                            <div style="margin-left: 20px;">
                            <p>- Case 1) 코드 1과 "disease1" 입력 후, 코드 2와 다시 "disease1" 입력: "disease1"이 포함된 행은 2로 코딩됩니다.</p>
                            <p>- Case 2) 코드 1과 "disease1" 입력 후, 코드 2와 "disease2" 입력: "disease1, disease2" 모두 포함된 행은 2로 코딩됩니다.</p>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.write(" ")
                    st.write(" ")
                    current_code = st.text_input("▶️ 코드를 입력하고 엔터를 누르세요: (ex - 0, 1, 2)", key="code_input")

                    if current_code:
                        current_code = int(current_code)  # Convert to integer code

                        # Check if current code already exists in session state
                        if current_code not in st.session_state.phrases_by_code:
                            st.session_state.phrases_by_code[current_code] = []  # Create a new list for this code if doesn't exist

                        # Define a callback function to handle text input
                        def add_text():
                            if st.session_state.text_input:
                                st.session_state.phrases_by_code[current_code].append(st.session_state.text_input)
                                st.session_state.text_input = ""  # Reset the input field

                        # Allow multiple text input with callback
                        st.text_input("▶️ 텍스트를 입력하고 엔터를 누르세요:", key="text_input", on_change=add_text)

                    # 4. 입력된 코딩 및 텍스트 목록 표시 및 삭제 기능 추가
                    if st.session_state.phrases_by_code:
                        st.write(" ")
                        st.write(" ")
                        st.markdown("<h5>현재 입력된 코드 및 텍스트 목록 :</h5>", unsafe_allow_html=True)
                        for code, phrases in st.session_state.phrases_by_code.items():
                            st.markdown(f"<strong><span style='color:#526E48;'>✅ 코드 {code}에 대한 텍스트:</span>", unsafe_allow_html=True)
                            # Create a dynamic list where phrases can be deleted
                            for phrase in phrases:
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"<div style='margin-left: 20px;'><span style='color:#AB4459;'>- {phrase}</span>", unsafe_allow_html=True)
                                with col2:
                                    if st.button(f"삭제", key=f"delete_{code}_{phrase}"):
                                        st.session_state.phrases_by_code[code].remove(phrase)  # Remove the phrase from the list
                                        # Force rerun by altering a session state value
                                        st.session_state["rerun_trigger"] = not st.session_state.get("rerun_trigger", False)

                    # 5. 미처리 항목을 자동으로 0으로 처리 또는 다른 방식 처리
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>코딩되지 않은 그 외 판독문 처리 방법</h4>", unsafe_allow_html=True)

                    # Use radio buttons to select between filling with 0 or missing
                    fill_option = st.radio("✔️ 처리 방법을 선택해주세요:", ("전부 0으로", "전부 99로", "전부 공백으로"))

                    # 3. 완료 버튼 - 텍스트 입력 후 활성화
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>코딩 작업 종료하기</h4>", unsafe_allow_html=True)
                    if current_code and st.session_state.phrases_by_code[current_code]:
                        if st.button("코딩 종료"):
                            # Create a temporary lowercase column for matching
                            df = st.session_state.df.copy()  # Use session_state to preserve df between runs
                            df['lower_temp'] = df[selected_column].str.lower()

                            # Process the text for each code
                            for code, phrases in st.session_state.phrases_by_code.items():
                                for phrase in phrases:
                                    # Match against the lowercase temporary column
                                    df['coding'] = df['coding'].where(~df['lower_temp'].str.contains(phrase.lower(), na=False), code)

                            # Apply the appropriate fill method based on the radio selection
                            if fill_option == "전부 0으로":
                                df['coding'].fillna(0, inplace=True)
                            elif fill_option == "전부 99로":
                                df['coding'].fillna(99, inplace=True)
                            elif fill_option == "전부 공백":
                                df['coding'].fillna(np.nan, inplace=True)

                            # Drop the temporary column after coding
                            df.drop(columns=['lower_temp'], inplace=True)

                            # Store the coded DataFrame in session state
                            st.session_state.coded_df = df

                            with st.spinner("Loading..."):
                                time.sleep(5)  # Simulate loading time

                            # Display coding result
                            st.success("코딩이 완료되었습니다. 결과를 확인하세요.", icon="✅")
                            st.dataframe(st.session_state.coded_df, use_container_width=True)

                        # 6. 데이터 다운로드 버튼 (Excel 또는 CSV)
                        if st.session_state.coded_df is not None:
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>코딩 데이터 다운로드</h4>", unsafe_allow_html=True)
                            export_format = st.radio("✔️ 파일 형식을 선택해주세요:", options=["CSV", "Excel"])
                            if export_format == "CSV":
                                csv = st.session_state.coded_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="CSV 다운로드",
                                    data=csv,
                                    file_name="bc_table.csv",
                                    mime='text/csv'
                                )
                            elif export_format == "Excel":
                                buffer = BytesIO()
                                try:
                                    # Use ExcelWriter with openpyxl
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        st.session_state.coded_df.to_excel(writer, index=False)

                                    # Move the buffer's position back to the start
                                    buffer.seek(0)

                                    # Offer the download button for Excel
                                    st.download_button(
                                        label="Excel 다운로드",
                                        data=buffer,
                                        file_name="bc_table.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                finally:
                                    buffer.close()

            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "📊 시각화":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📊 시각화</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;원하시는 데이터를 업로드하신 후 자유롭게 시각화하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write("")

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 시각화에 이용하실 데이터 파일을 업로드해주세요:")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="📊 시각화")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    st.divider()

                    # Select visualization format
                    st.header("📊 Univariable 데이터 시각화", divider='rainbow')
                    st.markdown(
                    """
                    <style>
                    .custom-callout {
                        background-color: #f9f9f9;
                        padding: 10px;  /* Adjust padding */
                        border-radius: 10px;
                        border: 1px solid #d3d3d3;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }
                    .custom-callout p {
                        margin: 0;
                        color: #000000;  /* Font color */
                        font-size: 14px;  /* Font size */
                        line-height: 1.4; /* Line height */
                        text-align: left; /* Align text to the left */
                    }
                    </style>
                    <div class="custom-callout">
                        <p>- 범주형 변수: 성별(남/여), 질환력(유/무)와 같이 고유한 값이나 범주 수가 제한된 변수</p>
                        <p>- 연속형 변수: 키, 몸무게, 혈액과 같이 숫자로 측정되며, 일정 범위 안에서 어떠한 값도 취할 수 있는 변수</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                    st.text("")
                    st.text("")
                    plot_type = st.radio("✔️ 그래프를 선택해주세요:", ('범주형 변수 : Barplot', '범주형 변수 : Pie chart', '연속형 변수 : Histogram', '연속형 변수 : Boxplot'))
                    st.text("")

                    # Creating visualizations using Plotlyif plot_type == '범주형 변수 : Barplot':
                    # Convert to categorical data if necessary
                    # Create visualizations based on user's selection
                    if plot_type:  # 사용자가 시각화 유형을 선택하면 실행
                        if plot_type == '범주형 변수 : Barplot':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")

                            selected_column = st.selectbox("✔️ 열을 선택해주세요:", columns)
                            if selected_column != "-- 선택 --":
                                if df[selected_column].dtype != 'category':
                                    df[selected_column] = df[selected_column].astype('category')
                                # Data preparation
                                count_data = df[selected_column].value_counts().reset_index()
                                count_data.columns = [selected_column, 'Count']  # Specify appropriate column names
                                # Create Barplot
                                fig = px.bar(count_data,
                                            x=selected_column,
                                            y='Count',
                                            labels={selected_column: selected_column, 'Count': 'Count'},
                                            color_discrete_sequence=["#FFBDBD", "#BBDDEE"])  # Specify color
                                st.plotly_chart(fig)

                        elif plot_type == '범주형 변수 : Pie chart':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")

                            selected_column = st.selectbox("✔️ 열을 선택해주세요:", columns)
                            if selected_column != "-- 선택 --":
                                if df[selected_column].dtype != 'category':
                                    df[selected_column] = df[selected_column].astype('category')
                                # Data preparation
                                count_data = df[selected_column].value_counts().reset_index()
                                count_data.columns = [selected_column, 'Count']  # Specify appropriate column names
                                # Create Barplot
                                fig = px.pie(
                                    count_data,
                                    names=selected_column,  # Categories for the pie slices
                                    values='Count',         # Values (count) for the pie slices
                                    labels={selected_column: selected_column, 'Count': 'Count'},  # Optional: Custom labels
                                    color_discrete_sequence=["#FFBDBD", "#BBDDEE"]  # Specify color
                                )

                                # Display the plot
                                st.plotly_chart(fig)

                        elif plot_type == '연속형 변수 : Histogram':
                            # Check if the selected column is continuous
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")

                            selected_column = st.selectbox("✔️ 열을 선택해주세요:", columns)
                            if selected_column != "-- 선택 --":
                                if df[selected_column].dtype in ['int64', 'float64']:
                                    fig = ff.create_distplot([df[selected_column].dropna()], [selected_column], bin_size=0.1)
                                    fig.update_layout(showlegend=False)
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Histogram은 연속형 변수에 적합합니다. 다른 열을 선택해주세요.")

                        elif plot_type == '연속형 변수 : Boxplot':
                            # Check if the selected column is continuous
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")

                            selected_column = st.selectbox("✔️ 열을 선택해주세요:", columns)
                            if selected_column != "-- 선택 --":
                                if df[selected_column].dtype in ['int64', 'float64']:
                                    fig = px.box(df, x=selected_column, color_discrete_sequence=["#BBDDEE"])  # Specify color
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Boxplot은 연속형 변수에 적합합니다. 다른 열을 선택해주세요.")

                        else:
                            st.write("먼저 시각화 유형을 선택해주세요.")

                    st.divider()

                        # Select visualization format
                    st.header("📊 Multivariable 데이터 시각화", divider='rainbow')
                    st.markdown(
                    """
                    <style>
                    .custom-callout {
                        background-color: #f9f9f9;
                        padding: 10px;  /* Adjust padding */
                        border-radius: 10px;
                        border: 1px solid #d3d3d3;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }
                    .custom-callout p {
                        margin: 0;
                        color: #000000;  /* Font color */
                        font-size: 14px;  /* Font size */
                        line-height: 1.4; /* Line height */
                        text-align: left; /* Align text to the left */
                    }
                    </style>
                    <div class="custom-callout">
                        <p>- 범주형 변수: 성별(남/여), 질환력(유/무)와 같이 고유한 값이나 범주 수가 제한된 변수</p>
                        <p>- 연속형 변수: 키, 몸무게, 혈액과 같이 숫자로 측정되며, 일정 범위 안에서 어떠한 값도 취할 수 있는 변수</p>
                        <p>⭐ 그룹열 변수는 범주형 변수로만 설정 가능합니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                    st.text("")
                    st.text("")
                    plot_type = st.radio("✔️ 그래프를 선택해주세요:", ('범주형 변수 : Barplot', '범주형 변수 : Pie chart', '연속형 변수 : Histogram', '연속형 변수 : Boxplot', '연속형 변수: Correlation Heatmap'))
                    st.text("")

                    # Creating visualizations using Plotlyif plot_type == '범주형 변수 : Barplot':
                    # Convert to categorical data if necessary
                    # Create visualizations based on user's selection
                    if plot_type:  # 사용자가 시각화 유형을 선택하면 실행
                        if plot_type == '범주형 변수 : Barplot':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("✔️ 열을 선택해주세요:", columns, key='categorical_variable')
                            selected_column_2 = st.selectbox("✔️ 그룹열을 선택해주세요:", columns, key='group_variable')

                            if selected_column_1 != "-- 선택 --":
                                if df[selected_column_1].dtype != 'category':
                                    df[selected_column_1] = df[selected_column_1].astype('category')
                            if selected_column_2 != "-- 선택 --":
                                if df[selected_column_2].dtype != 'category':
                                    df[selected_column_2] = df[selected_column_2].astype('category')

                                # Create Barplot
                                    # Data preparation: Group by selected_column_2 and count selected_column_1
                                count_data = df.groupby([selected_column_2, selected_column_1]).size().reset_index(name='Count')
                                count_data = count_data.sort_values(by=selected_column_2)

                                # Create Barplot
                                fig = px.bar(
                                    count_data,
                                    x=selected_column_1,
                                    y='Count',
                                    color=selected_column_2,
                                    barmode='group',  # Change to 'group' for side-by-side bars
                                    labels={selected_column_1: selected_column_1, 'Count': 'Count'},
                                    color_discrete_sequence=px.colors.qualitative.Set2  # Use a qualitative color palette
                                )

                                # Hide the legend
                                fig.update_layout(showlegend=True)  # Hides the legend

                                # Display the plot
                                st.plotly_chart(fig)

                        elif plot_type == '범주형 변수 : Pie chart':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("✔️ 열을 선택해주세요:", columns, key='categorical_variable')
                            selected_column_2 = st.selectbox("✔️ 그룹열을 선택해주세요:", columns, key='group_variable')

                            if selected_column_1 != "-- 선택 --" and selected_column_2 != "-- 선택 --":
                                # Data preparation: Group by selected_column_2 and count selected_column_1
                                count_data = df.groupby([selected_column_2, selected_column_1]).size().reset_index(name='Count')
                                count_data = count_data.sort_values(by=selected_column_2)

                                # Get unique values in selected_column_2 for separate pie charts
                                unique_groups = count_data[selected_column_2].unique()

                                # Create a pie chart for each unique group in selected_column_2
                                for group in unique_groups:
                                    group_data = count_data[count_data[selected_column_2] == group]

                                    # Create Pie Chart
                                    fig = px.pie(
                                        group_data,
                                        names=selected_column_1,  # Categories for the pie slices
                                        values='Count',            # Values for the pie chart
                                        title=f"{selected_column_2}열의 {group}에 대한 파이 차트",  # Title indicating the group
                                        color_discrete_sequence=px.colors.qualitative.Set2  # Use a qualitative color palette
                                    )

                                    # Display the plot
                                    st.plotly_chart(fig)

                        elif plot_type == '연속형 변수 : Histogram':
                            # Check if the selected column is continuous
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("✔️ 열을 선택해주세요:", columns, key='continuous_variable')
                            selected_column_2 = st.selectbox("✔️ 그룹열을 선택해주세요:", columns, key='group_variable')

                            if selected_column_1 != "-- 선택 --" and selected_column_2 != "-- 선택 --":
                                # Check if selected_column_1 is continuous
                                if df[selected_column_1].dtype in ['int64', 'float64']:
                                    # Check if selected_column_2 is categorical
                                    if df[selected_column_2].dtype != 'category':
                                        df[selected_column_2] = df[selected_column_2].astype('category')

                                    # Prepare data for distplot
                                    # Get sorted unique categories from selected_column_2
                                    sorted_categories = sorted(df[selected_column_2].cat.categories)

                                    data_to_plot = [
                                        df[df[selected_column_2] == category][selected_column_1].dropna().tolist()
                                        for category in sorted_categories
                                    ]

                                    # Create Distplot
                                    fig = ff.create_distplot(data_to_plot,
                                                            group_labels=sorted_categories,  # Use sorted categories for labels
                                                            bin_size=0.1)

                                    # Display the plot
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Histogram은 연속형 변수에 적합합니다. 다른 열을 선택해주세요:")

                        elif plot_type == '연속형 변수 : Boxplot':
                            # Check if the selected column is continuous
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("✔️ 열을 선택해주세요:", columns, key='continuous_variable')
                            selected_column_2 = st.selectbox("✔️ 그룹열을 선택해주세요:", columns, key='group_variable')

                            if selected_column_1 != "-- 선택 --" and selected_column_2 != "-- 선택 --":
                                # Check if selected_column_1 is continuous
                                if df[selected_column_1].dtype in ['int64', 'float64']:
                                    # Check if selected_column_2 is categorical
                                    if df[selected_column_2].dtype != 'category':
                                        df[selected_column_2] = df[selected_column_2].astype('category')

                                    sorted_categories = df[selected_column_2].cat.categories.tolist() if df[selected_column_2].dtype == 'category' else df[selected_column_2].unique().tolist()
                                    sorted_categories.sort()

                                    # Create Box Plot
                                    fig = px.box(
                                        df,
                                        x=selected_column_2,
                                        y=selected_column_1,
                                        color=selected_column_2,  # Color by selected_column_2
                                        color_discrete_sequence=px.colors.qualitative.Set2,  # Use a qualitative color palette
                                        title='Box Plot of {} by {}'.format(selected_column_1, selected_column_2),
                                        labels={selected_column_2: selected_column_2, selected_column_1: selected_column_1}  # Custom labels
                                    )

                                    # Sort the x-axis categories based on sorted_categories
                                    fig.update_layout(xaxis=dict(categoryorder='array', categoryarray=sorted_categories))

                                    # Display the plot
                                    st.plotly_chart(fig)
                                else:
                                    st.warning("Boxplot은 연속형 변수에 적합합니다. 다른 열을 선택해주세요:")

                        elif plot_type == '연속형 변수: Correlation Heatmap':
                            # Select numerical columns for correlation
                            numeric_columns = [col for col in df.select_dtypes(include=['int64', 'float64']).columns if df[col].nunique() > 10]
                            if len(numeric_columns) > 1:
                                st.warning("히트맵은 연속형 변수만 이용 가능합니다. \n 고유 값이 10개 이하인 변수는 범주형 변수로 간주하여 자동 제외됩니다.", icon="🚨")
                                corr = df[numeric_columns].corr()

                                # Plotly interactive correlation matrix
                                fig = px.imshow(round(corr, 3), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
                                st.plotly_chart(fig)
                            else:
                                st.warning("히트맵을 구현할 연속형 변수가 충분하지 않습니다.")

                        else:
                            st.write("먼저 시각화 유형을 선택해주세요:")

            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "📊 특성표 생성":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📊 특성표 생성</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;원하시는 데이터를 업로드하신 후 자유롭게 분석하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 특성표 생성에 이용하실 데이터 파일을 업로드해주세요:")
        st.warning("업로드 시, 날짜형 타입의 열은 자동으로 인식하여 제외됩니다.", icon="🚨")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="📊 특성표 생성")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    st.divider()

                    # Select visualization format
                    st.markdown("<h4 style='color:grey;'>데이터 N수 파악</h4>", unsafe_allow_html=True)
                    st.markdown(
                    """
                    <style>
                    .custom-callout {
                        background-color: #f9f9f9;
                        padding: 10px;  /* Adjust padding */
                        border-radius: 10px;
                        border: 1px solid #d3d3d3;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }
                    .custom-callout p {
                        margin: 0;
                        color: #000000;  /* Font color */
                        font-size: 14px;  /* Font size */
                        line-height: 1.4; /* Line height */
                        text-align: left; /* Align text to the left */
                    }
                    </style>
                    <div class="custom-callout">
                        <p>- 범주형 변수는 unique 값이 3개 이하인 경우로만 탐색됩니다.</p>
                        <p>- 범주형 변수에 대해선 n(percentage) 형태가, 연속형 변수에 대해선 mean[IQR] 형태가 도출됩니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

                    def is_date_column(column):
                        """Function to determine if a column should be considered as a date column."""
                        return pd.api.types.is_datetime64_any_dtype(column) or pd.api.types.is_object_dtype(column) or column.name.lower().endswith("date")

                    def bs_count(sample):
                        # Categorical columns with 2 or 3 unique values, excluding potential date columns
                        cat_col = [col for col in sample.columns if sample[col].nunique() <= 3 and not is_date_column(sample[col])]

                        # Two unique values
                        cat_table2 = pd.DataFrame()
                        cat_col2 = [col for col in cat_col if sample[col].nunique() == 2]

                        for col in cat_col2:
                            counts = sample[col].value_counts().sort_index()
                            percentages = (sample[col].value_counts(normalize=True) * 100).round(1).sort_index()

                            for level in counts.index:
                                var_name = f"{col}_{level}"
                                count_value = counts[level]
                                percent_value = percentages[level]
                                merged_value = f"{count_value:,.0f} ({percent_value:.1f})"
                                cat_table2 = pd.concat([cat_table2, pd.DataFrame({'Variable': [var_name], 'Count': [merged_value]})], ignore_index=True)

                        # Three unique values
                        cat_table3 = pd.DataFrame()
                        cat_col3 = [col for col in cat_col if sample[col].nunique() == 3 and not is_date_column(sample[col])]

                        for col in cat_col3:
                            counts = sample[col].value_counts().sort_index()
                            percentages = (sample[col].value_counts(normalize=True) * 100).round(1).sort_index()

                            for level in counts.index:
                                var_name = f"{col}_{level}"
                                count_value = counts[level]
                                percent_value = percentages[level]
                                merged_value = f"{count_value:,.0f} ({percent_value:.1f})"
                                cat_table3 = pd.concat([cat_table3, pd.DataFrame({'Variable': [var_name], 'Count': [merged_value]})], ignore_index=True)

                        # Numerical values
                        num_table = pd.DataFrame(columns=['Variable', 'Count'])
                        num_col = [col for col in sample.columns if col not in cat_col and not is_date_column(sample[col])]

                        for col in num_col:
                            means = sample[col].mean()
                            iqr = sample[col].quantile(0.75) - sample[col].quantile(0.25)
                            num_table = pd.concat([num_table, pd.DataFrame({'Variable': [col], 'Count': ['{:,.1f} [{:,.1f}]'.format(means, iqr)]})], ignore_index=True)

                        # Combine all tables
                        final_tab = pd.concat([cat_table2, cat_table3, num_table], axis=0, ignore_index=True)

                        # Extract original column names without levels for merging
                        # for col in cat_col2 + cat_col3:
                            # final_tab.loc[final_tab['Variable'].str.startswith(col), 'Base_Col'] = final_tab['Variable'].str.rsplit('_',n=1).str[0]

                        # num_col의 컬럼들에 대해, Variable 값 자체를 Base_Col로 유지
                        # for col in num_col:
                            # final_tab.loc[final_tab['Variable'] == col, 'Base_Col'] = col

                        return final_tab


                    def bs_res_count(sample, response_col):
                        # Ensure the response column is categorical with only two categories
                        if sample[response_col].dtype.name != 'category':
                            sample[response_col] = sample[response_col].astype('category')

                        if sample[response_col].nunique() != 2:
                            raise ValueError("선택할 종속변수가 2개의 범주를 가져야합니다.")

                        # Remove rows with missing response column values
                        sample = sample.dropna(subset=[response_col])

                        # Identify categorical columns based on the number of unique values and the data type
                        cat_col = [col for col in sample.columns if (sample[col].nunique() <= 3) and not is_date_column(sample[col])]
                        for col in cat_col:
                            sample[col] = sample[col].astype('category')

                        if response_col in cat_col:
                            cat_col.remove(response_col)

                        # Analysis for categorical columns
                        results_cat = []
                        for col in cat_col:
                            # Drop rows with missing values in the column
                            cat_sample = sample.dropna(subset=[col])

                            # Calculate counts and percentages grouped by the response column
                            counts = cat_sample.groupby([response_col, col]).size().unstack(fill_value=0)
                            percentages = counts.div(counts.sum(axis=1), axis=0) * 100

                            # Create a combined table with count and percentage
                            for level in counts.columns:
                                row_data = {
                                    'Variable': f"{col}_{level}",
                                    # 'Statistic': None,
                                    'P-value': None
                                }
                                for group in counts.index:
                                    count_value = counts.loc[group, level]
                                    percent_value = percentages.loc[group, level]
                                    row_data[f'{response_col}_{group}'] = f"{count_value:,} ({percent_value:.1f})"
                                results_cat.append(row_data)

                            # Chi-square test for association between the column and response variable
                            crosstab = pd.crosstab(cat_sample[col], cat_sample[response_col], margins=False)
                            if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:  # Ensure at least 2x2 table
                                chi2_stat, chi2_p = chi2_contingency(observed=crosstab, correction=False)[:2]
                                for row in results_cat:
                                    if row['Variable'].startswith(col):
                                        # row['Statistic'] = round(chi2_stat, 3)
                                        row['P-value'] = '<0.001' if chi2_p < 0.001 else round(chi2_p, 3)

                        # Analysis for numerical columns
                        results_num = []
                        num_col = [col for col in sample.columns if col not in cat_col + [response_col] and not is_date_column(sample[col])]
                        for col in num_col:
                            # Drop rows with missing values in the numerical column
                            num_sample = sample.dropna(subset=[col])

                            # Test for normality across both groups using Shapiro-Wilk test
                            normality_results = []
                            for category in num_sample[response_col].cat.categories:
                                group_data = num_sample[num_sample[response_col] == category][col].dropna()
                                if len(group_data) >= 3:
                                    normality_results.append(shapiro(group_data)[1])
                                else:
                                    normality_results.append(0)  # Consider non-normal if the group has less than 3 observations

                            is_normal = all(p > 0.05 for p in normality_results)

                            # Calculate mean and IQR grouped by the response column
                            group_stats = num_sample.groupby(response_col).agg(
                                mean=(col, 'mean'),
                                q1=(col, lambda x: x.quantile(0.25)),
                                q3=(col, lambda x: x.quantile(0.75))
                            )
                            group_stats['iqr'] = group_stats['q3'] - group_stats['q1']

                            row_data = {
                                'Variable': col,
                                # 'Statistic': None,
                                'P-value': None
                            }
                            for group in group_stats.index:
                                mean_value = group_stats.loc[group, 'mean']
                                iqr_value = group_stats.loc[group, 'iqr']
                                row_data[f'{response_col}_{group}'] = f"{mean_value:.1f} [{iqr_value:.1f}]"
                            results_num.append(row_data)

                            # Perform the appropriate test based on normality
                            groups = [num_sample[num_sample[response_col] == category][col].dropna() for category in num_sample[response_col].cat.categories]
                            if len(groups) == 2:  # Ensure there are exactly two groups for t-test
                                if is_normal:
                                    # Check for equal variances using Levene's test
                                    _, p_levene = levene(*groups)
                                    equal_var = p_levene > 0.05

                                    # Perform t-test if data is normally distributed
                                    t_stat, t_p = ttest_ind(*groups, equal_var=equal_var)
                                    # row_data['Statistic'] = round(t_stat, 3)
                                    row_data['P-value'] = '<0.001' if t_p < 0.001 else round(t_p, 3)
                                else:
                                    # Perform Mann-Whitney U test if data is not normally distributed
                                    u_stat, u_p = kruskal(*groups)  # Using Kruskal-Wallis with 2 groups as Mann-Whitney U alternative
                                    # row_data['Statistic'] = round(u_stat, 3)
                                    row_data['P-value'] = '<0.001' if u_p < 0.001 else round(u_p, 3)

                        # Combine results and create a DataFrame
                        final_results = pd.DataFrame(results_cat + results_num)

                        # Reorder columns to match the desired format
                        response_cols = [col for col in final_results.columns if col.startswith(response_col)]
                        final_results = final_results[['Variable'] + response_cols + ['P-value']]

                        return final_results

                    def bs_res_count3(sample, response_col):
                        # Ensure the response column is categorical with three categories
                        if sample[response_col].dtype.name != 'category':
                            sample[response_col] = sample[response_col].astype('category')

                        if sample[response_col].nunique() != 3:
                            raise ValueError("선택할 종속변수가 3개의 범주를 가져야합니다.")

                        # Remove rows with missing response column values
                        sample = sample.dropna(subset=[response_col])

                        # Identify categorical columns based on the number of unique values and the data type
                        cat_col = [col for col in sample.columns if (sample[col].nunique() <= 3) and not is_date_column(sample[col])]
                        for col in cat_col:
                            sample[col] = sample[col].astype('category')

                        if response_col in cat_col:
                            cat_col.remove(response_col)

                        # Analysis for categorical columns
                        results_cat = []
                        for col in cat_col:
                            # Drop rows with missing values in the column
                            cat_sample = sample.dropna(subset=[col])

                            # Calculate counts and percentages grouped by the response column
                            counts = cat_sample.groupby([response_col, col]).size().unstack(fill_value=0)
                            percentages = counts.div(counts.sum(axis=1), axis=0) * 100

                            # Create a combined table with count and percentage
                            for level in counts.columns:
                                row_data = {
                                    'Variable': f"{col}_{level}",
                                    'P-value': None
                                }
                                for group in counts.index:
                                    count_value = counts.loc[group, level]
                                    percent_value = percentages.loc[group, level]
                                    row_data[f'{response_col}_{group}'] = f"{count_value:,} ({percent_value:.1f})"
                                results_cat.append(row_data)

                            # Chi-square test for association between the column and response variable
                            crosstab = pd.crosstab(cat_sample[col], cat_sample[response_col], margins=False)
                            if crosstab.shape[0] > 1 and crosstab.shape[1] > 1:  # Ensure at least 2x3 table
                                chi2_stat, chi2_p = chi2_contingency(observed=crosstab, correction=False)[:2]
                                for row in results_cat:
                                    if row['Variable'].startswith(col):
                                        row['P-value'] = '<0.001' if chi2_p < 0.001 else round(chi2_p, 3)

                        # Analysis for numerical columns
                        results_num = []
                        num_col = [col for col in sample.columns if col not in cat_col + [response_col] and not is_date_column(sample[col])]
                        for col in num_col:
                            # Drop rows with missing values in the numerical column
                            num_sample = sample.dropna(subset=[col])

                            # Test for normality across all groups using Shapiro-Wilk test
                            normality_results = []
                            for category in num_sample[response_col].cat.categories:
                                group_data = num_sample[num_sample[response_col] == category][col].dropna()
                                if len(group_data) >= 3:
                                    normality_results.append(shapiro(group_data)[1])
                                else:
                                    normality_results.append(0)  # Consider non-normal if the group has less than 3 observations

                            is_normal = all(p > 0.05 for p in normality_results)

                            # Calculate mean and IQR grouped by the response column
                            group_stats = num_sample.groupby(response_col).agg(
                                mean=(col, 'mean'),
                                q1=(col, lambda x: x.quantile(0.25)),
                                q3=(col, lambda x: x.quantile(0.75))
                            )
                            group_stats['iqr'] = group_stats['q3'] - group_stats['q1']

                            row_data = {
                                'Variable': col,
                                'P-value': None
                            }
                            for group in group_stats.index:
                                mean_value = group_stats.loc[group, 'mean']
                                iqr_value = group_stats.loc[group, 'iqr']
                                row_data[f'{response_col}_{group}'] = f"{mean_value:.1f} [{iqr_value:.1f}]"
                            results_num.append(row_data)

                            # Perform the appropriate test based on normality
                            groups = [num_sample[num_sample[response_col] == category][col].dropna() for category in num_sample[response_col].cat.categories]
                            if len(groups) == 3:  # Ensure there are exactly three groups for ANOVA or Kruskal-Wallis
                                if is_normal:
                                    # Check for equal variances using Levene's test
                                    _, p_levene = levene(*groups)
                                    equal_var = p_levene > 0.05

                                    # Perform ANOVA if data is normally distributed
                                    f_stat, anova_p = f_oneway(*groups)
                                    row_data['P-value'] = '<0.001' if anova_p < 0.001 else round(anova_p, 3)
                                else:
                                    # Perform Kruskal-Wallis test if data is not normally distributed
                                    kw_stat, kw_p = kruskal(*groups)
                                    row_data['P-value'] = '<0.001' if kw_p < 0.001 else round(kw_p, 3)

                        # Combine results and create a DataFrame
                        final_results = pd.DataFrame(results_cat + results_num)

                        # Reorder columns to match the desired format
                        response_cols = [col for col in final_results.columns if col.startswith(response_col)]
                        final_results = final_results[['Variable'] + response_cols + ['P-value']]

                        return final_results


                    # Function to merge bs_count and bs_char
                    def merge_results(count_result, char_result):
                        # Merge based on the last occurrence of '_'
                        merged_result = pd.merge(count_result, char_result, how='left', on='Variable')
                        # Drop the auxiliary 'Base_Col' used for merging
                        return merged_result

                    st.write(" ")
                    count = bs_count(df)
                    st.dataframe(count[['Variable', 'Count']], use_container_width=True)

                    st.divider()
                    st.header("📊 특성표 생성", divider="rainbow")

                    # Let the user select whether the dependent variable has 2 or 3 categories
                    category_choice = st.radio(
                        "✔️ 종속변수가 몇 개의 범주를 가지는지 선택해주세요:",
                        options=["2 범주", "3 범주"]
                    )
                    st.write()

                    if category_choice == "2 범주":
                        st.markdown(
                        """
                        <style>
                        .custom-callout {
                            background-color: #f9f9f9;
                            padding: 10px;  /* Adjust padding */
                            border-radius: 10px;
                            border: 1px solid #d3d3d3;
                            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        }
                        .custom-callout p {
                            margin: 0;
                            color: #000000;  /* Font color */
                            font-size: 14px;  /* Font size */
                            line-height: 1.4; /* Line height */
                            text-align: left; /* Align text to the left */
                        }
                        </style>
                        <div class="custom-callout">
                            <p>- 종속변수가 결측인 행은 제외 후 count됩니다.</p>
                            <p>- 등분산성을 갖는 연속형 변수에 Student t-검정이 사용되며, 이분산성을 갖는 연속형 변수에 Welch's t-검정이 적용됩니다.</p>
                            <p>- 범주형 변수에 Chi-square 검정이 사용됩니다.</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                        )
                        st.write(" ")
                        st.write(" ")

                        response_col = st.selectbox(
                            "✔️ 통계적 유의성을 볼 종속변수를 선택해주세요:",
                            options=["-- 선택 --"] + [col for col in df.columns if df[col].nunique() == 2],
                            index=0
                        )

                        if response_col != "-- 선택 --":
                            char = bs_res_count(df, response_col)
                            st.dataframe(char, use_container_width=True)

                            st.divider()
                            st.markdown("<h4 style='color:grey;'>최종 특성표</h4>", unsafe_allow_html=True)
                            merged = merge_results(count, char)
                            st.dataframe(merged, use_container_width=True)

                            st.divider()
                            st.markdown("<h4 style='color:grey;'>특성표 다운로드</h4>", unsafe_allow_html=True)
                            export_format = st.radio("✔️ 파일 형식을 선택해주세요:", options=["CSV", "Excel"])
                            if export_format == "CSV":
                                csv = merged.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="CSV 다운로드",
                                    data=csv,
                                    file_name="bc_table.csv",
                                    mime='text/csv'
                                )
                            elif export_format == "Excel":
                                buffer = BytesIO()
                                try:
                                    # Use ExcelWriter with openpyxl
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        merged.to_excel(writer, index=False)

                                    # Move the buffer's position back to the start
                                    buffer.seek(0)

                                    # Offer the download button for Excel
                                    st.download_button(
                                        label="Excel 다운로드",
                                        data=buffer,
                                        file_name="bc_table.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                finally:
                                    buffer.close()
                        else:
                            st.write("")

                    elif category_choice == "3 범주":
                        st.markdown(
                        """
                        <style>
                        .custom-callout {
                            background-color: #f9f9f9;
                            padding: 10px;  /* Adjust padding */
                            border-radius: 10px;
                            border: 1px solid #d3d3d3;
                            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                        }
                        .custom-callout p {
                            margin: 0;
                            color: #000000;  /* Font color */
                            font-size: 14px;  /* Font size */
                            line-height: 1.4; /* Line height */
                            text-align: left; /* Align text to the left */
                        }
                        </style>
                        <div class="custom-callout">
                            <p>- 종속변수가 결측인 행은 제외 후 count됩니다.</p>
                            <p>- 등분산성을 갖는 연속형 변수에 ANOVA 검정이 사용되며, 이분산성을 갖는 연속형 변수에 Kruskal-Wallis 검정이 적용됩니다.</p>
                            <p>- 범주형 변수에 Chi-square 검정이 사용됩니다.</p>

                        </div>
                        """,
                        unsafe_allow_html=True
                        )
                        st.write(" ")
                        st.write(" ")

                        response_col = st.selectbox(
                            "✔️ 통계적 유의성을 볼 종속변수를 선택해주세요:",
                            options=["-- 선택 --"] + [col for col in df.columns if df[col].nunique() == 3],
                            index=0
                        )

                        if response_col != "-- 선택 --":
                            char = bs_res_count3(df, response_col)
                            st.dataframe(char, use_container_width=True)

                            st.divider()
                            st.markdown("<h4 style='color:grey;'>최종 특성표</h4>", unsafe_allow_html=True)
                            merged = merge_results(count, char)
                            st.dataframe(merged, use_container_width=True)

                            st.divider()
                            st.markdown("<h4 style='color:grey;'>특성표 다운로드</h4>", unsafe_allow_html=True)
                            export_format = st.radio("✔️ 파일 형식을 선택해주세요:", options=["CSV", "Excel"])
                            if export_format == "CSV":
                                csv = merged.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="CSV 다운로드",
                                    data=csv,
                                    file_name="bc_table.csv",
                                    mime='text/csv'
                                )
                            elif export_format == "Excel":
                                buffer = BytesIO()
                                try:
                                    # Use ExcelWriter with openpyxl
                                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                        merged.to_excel(writer, index=False)

                                    # Move the buffer's position back to the start
                                    buffer.seek(0)

                                    # Offer the download button for Excel
                                    st.download_button(
                                        label="Excel 다운로드",
                                        data=buffer,
                                        file_name="bc_table.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                finally:
                                    buffer.close()
                        else:
                            st.write("")
            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "♻️ 인과관계 추론":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">♻️ 인과관계 추론</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;분석 전 데이터 인과관계를 파악해보세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # Track the uploaded file in session state to reset the UI when a new file is uploaded
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📁 인과관계를 볼 데이터 파일을 업로드해주세요:", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                # 데이터 미리보기 표시
                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="♻️ 인과관계 추론")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    # st.session_state 초기화
                    if "continuous_columns" not in st.session_state:
                        st.session_state.continuous_columns = []
                    if "categorical_columns" not in st.session_state:
                        st.session_state.categorical_columns = []
                    if "proceed_to_preprocessing" not in st.session_state:
                        st.session_state.proceed_to_preprocessing = False
                    if "causal_inference_ready" not in st.session_state:
                        st.session_state.causal_inference_ready = False
                    if "causal_inference_triggered" not in st.session_state:
                        st.session_state.causal_inference_triggered = False
                    if "random_seed" not in st.session_state:
                        st.session_state.random_seed = 111
                    if "causal_graph_rendered" not in st.session_state:
                        st.session_state.causal_graph_rendered = False

                    # 판독문 열 선택창
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>변수 선택</h4>", unsafe_allow_html=True)
                    st.write("▶️ 인과관계를 볼 변수 열 선택")

                    # 선택된 열을 기반으로 작업
                    st.session_state.df = df  # Ensure df is stored initially

                    # 범주형 변수 추출 함수
                    def get_categorical_columns(df):
                        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
                        low_cardinality_numerical = [
                            col for col in df.select_dtypes(exclude=['object', 'category']).columns if df[col].nunique() < 5
                        ]
                        return categorical_columns + low_cardinality_numerical

                    # 연속형 변수 선택
                    continuous_columns = st.multiselect(
                        "✔️ 연속형 변수를 선택해주세요:",
                        df.select_dtypes(include=['float64', 'int64']).columns,
                        key="continuous_columns_selection"
                    )

                    # 범주형 변수 선택
                    categorical_columns = st.multiselect(
                        "✔️ 범주형 변수를 선택해주세요:",
                        get_categorical_columns(df),
                        key="categorical_columns_selection"
                    )

                    # 선택한 변수 기록
                    st.session_state.X_columns = continuous_columns + categorical_columns

                    # 선택 완료 버튼
                    if st.button('🚀 선택 완료', key='complete_button'):
                        if len(continuous_columns) + len(categorical_columns) > 1:
                            st.session_state.continuous_columns = continuous_columns
                            st.session_state.categorical_columns = categorical_columns
                            st.session_state.proceed_to_preprocessing = True
                            st.success("변수 선택이 완료되었습니다. 다음 단계로 진행하세요.", icon="✅")
                        else:
                            st.warning("변수를 두 개 이상 선택하셔야 합니다.", icon="⚠️")

                    # 2. 전처리 단계
                    if st.session_state.get("proceed_to_preprocessing", False):
                        st.divider()
                        st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                        # 데이터 분리
                        X_continuous = df[st.session_state.continuous_columns]
                        X_categorical = df[st.session_state.categorical_columns]

                        # 결측값 유무 확인
                        continuous_missing = X_continuous.isnull().any().any()
                        categorical_missing = X_categorical.isnull().any().any()

                        # 결측값 처리 필요 여부 확인
                        if not continuous_missing and not categorical_missing:
                            st.success("결측 처리 작업 없이 분석이 가능합니다.", icon="✅")
                            st.session_state.causal_inference_ready = True  # 바로 관계 추론 가능
                        else:
                            # 결측 처리 로직
                            continuous_missing_value_strategies = {}
                            categorical_missing_value_strategies = {}

                            # 연속형 변수 결측 처리 선택
                            for column in st.session_state.continuous_columns:
                                missing_count = df[column].isna().sum()
                                if missing_count > 0:
                                    st.error(f"선택하신 연속형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                    strategy = st.selectbox(
                                        f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                        ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                        key=f"continuous_{column}_strategy"
                                    )
                                    if strategy != '-- 선택 --':
                                        continuous_missing_value_strategies[column] = strategy

                            # 범주형 변수 결측 처리 선택
                            for column in st.session_state.categorical_columns:
                                missing_count = df[column].isna().sum()
                                if missing_count > 0:
                                    st.error(f"선택하신 범주형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                    strategy = st.selectbox(
                                        f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                        ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                        key=f"categorical_{column}_strategy"
                                    )
                                    if strategy != '-- 선택 --':
                                        categorical_missing_value_strategies[column] = strategy

                            # 결측 처리 버튼
                            if st.button("결측 처리"):
                                for column, strategy in continuous_missing_value_strategies.items():
                                    if strategy == '결측이 존재하는 행을 제거':
                                        X_continuous = X_continuous.dropna(subset=[column])
                                    else:
                                        impute_strategy = {
                                            '해당 열의 평균값으로 대체': 'mean',
                                            '해당 열의 중앙값으로 대체': 'median',
                                            '해당 열의 최빈값으로 대체': 'most_frequent'
                                        }[strategy]
                                        imputer = SimpleImputer(strategy=impute_strategy)
                                        X_continuous[[column]] = imputer.fit_transform(X_continuous[[column]])

                                for column, strategy in categorical_missing_value_strategies.items():
                                    if strategy == '결측이 존재하는 행을 제거':
                                        X_categorical = X_categorical.dropna(subset=[column])
                                    elif strategy == '해당 열의 최빈값으로 대체':
                                        imputer = SimpleImputer(strategy='most_frequent')
                                        X_categorical[[column]] = imputer.fit_transform(X_categorical[[column]])

                                # 인덱스 동기화
                                shared_indexes = X_continuous.index.intersection(X_categorical.index)
                                X_continuous = X_continuous.loc[shared_indexes]
                                X_categorical = X_categorical.loc[shared_indexes]

                                st.success("결측값 처리가 완료되었습니다. 분석을 진행하세요.", icon="✅")
                                st.session_state.causal_inference_ready = True

                    # 인과관계 추론 섹션 표시
                    if st.session_state.get("causal_inference_ready", False):
                        st.divider()
                        st.header("♻️ 인과관계 추론", divider="rainbow")

                        # 데이터 결합
                        if not X_continuous.empty or not X_categorical.empty:
                            X = pd.concat([X_continuous, X_categorical], axis=1)

                            # 데이터 전처리
                            preprocessor = ColumnTransformer(
                                transformers=[
                                    ('num', StandardScaler(), list(X_continuous.columns)),  # 연속형 변수 표준화
                                    ('cat', OneHotEncoder(drop='first'), list(X_categorical.columns))  # 범주형 변수 원-핫 인코딩
                                ]
                            )
                            X_transformed = preprocessor.fit_transform(X)

                            # Transformed column names (to align with X_transformed)
                            transformed_columns = (
                                list(X_continuous.columns) +  # Keep continuous column names
                                list(preprocessor.named_transformers_['cat'].get_feature_names_out(X_categorical.columns))  # Extract new column names for categorical variables
                            )

                            # Streamlit 컨테이너를 생성하여 그래프를 출력
                            causal_graph_container = st.container()

                            # PC 알고리즘 실행
                            cg = pc(X_transformed, alpha=0.05)

                            # Check if graph size matches transformed column names
                            if len(transformed_columns) != len(cg.G.graph):
                                raise ValueError(
                                    f"Mismatch between graph nodes ({len(cg.G.graph)}) and transformed columns ({len(transformed_columns)})."
                                )

                            # 방향성 있는 인과관계만 추출
                            def extract_directed_edges(causal_graph, column_names):
                                edges = []
                                for i in range(len(causal_graph)):
                                    for j in range(len(causal_graph)):
                                        if causal_graph[i, j] == 1 and causal_graph[j, i] != 1:  # Only i → j
                                            edges.append((column_names[i], column_names[j]))
                                        elif causal_graph[i, j] == -1 and causal_graph[j, i] != -1:  # Only j → i
                                            edges.append((column_names[j], column_names[i]))
                                return edges

                            # Extract directed edges using transformed column names
                            edges = extract_directed_edges(cg.G.graph, transformed_columns)

                            # Create the causal graph
                            causal_graph = nx.DiGraph()
                            causal_graph.add_edges_from(edges)

                            # 그래프 시각화 함수
                            def visualize_graph(graph, seed=None, padding_ratio=0.1, node_separation=1.5):
                                pos = nx.spring_layout(graph, seed=seed, k=node_separation, iterations=100)

                                edge_x = []
                                edge_y = []
                                annotations = []

                                for edge in graph.edges():
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    dx, dy = x1 - x0, y1 - y0
                                    dist = (dx**2 + dy**2)**0.5

                                    # 패딩 적용
                                    x0_padded = x0 + dx * padding_ratio / dist
                                    y0_padded = y0 + dy * padding_ratio / dist
                                    x1_padded = x1 - dx * padding_ratio / dist
                                    y1_padded = y1 - dy * padding_ratio / dist

                                    edge_x.extend([x0_padded, x1_padded, None])
                                    edge_y.extend([y0_padded, y1_padded, None])

                                    annotations.append(
                                        dict(
                                            ax=x0_padded, ay=y0_padded,
                                            x=x1_padded, y=y1_padded,
                                            xref="x", yref="y",
                                            axref="x", ayref="y",
                                            showarrow=True,
                                            arrowhead=3,
                                            arrowsize=1.5,
                                            arrowwidth=1.5,
                                            arrowcolor="gray"
                                        )
                                    )

                                node_x = []
                                node_y = []
                                node_text = []
                                for node in graph.nodes():
                                    x, y = pos[node]
                                    node_x.append(x)
                                    node_y.append(y)
                                    node_text.append(node)

                                node_trace = go.Scatter(
                                    x=node_x,
                                    y=node_y,
                                    mode='markers+text',
                                    text=node_text,
                                    textfont=dict(family='Times New Roman', size=12, color='black'),
                                    marker=dict(
                                        size=60,
                                        color='lightblue',
                                        line=dict(width=2, color='darkblue')
                                    ),
                                    hoverinfo='text'
                                )

                                edge_trace = go.Scatter(
                                    x=edge_x,
                                    y=edge_y,
                                    line=dict(width=1.5, color='gray'),
                                    hoverinfo='none',
                                    mode='lines'
                                )

                                fig = go.Figure(data=[edge_trace, node_trace])
                                fig.update_layout(
                                    height=800,
                                    showlegend=False,
                                    title_text="Causal Graph",
                                    title_font=dict(family="Times New Roman", size=20, color="black"),
                                    margin=dict(l=50, r=50, t=50, b=50),
                                    xaxis=dict(showgrid=False, zeroline=False),
                                    yaxis=dict(showgrid=False, zeroline=False),
                                    annotations=annotations
                                )

                                return fig

                            # Graph rendering
                            regenerate_layout_clicked = st.button("Figure 생성", key="regenerate_causal_layout_button")

                            if regenerate_layout_clicked or not st.session_state.causal_graph_rendered:
                                if regenerate_layout_clicked:
                                    st.session_state.random_seed = np.random.randint(0, 9999)

                                fig = visualize_graph(causal_graph, seed=st.session_state.random_seed)  # Pass `causal_graph` explicitly
                                st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_causal_{st.session_state.random_seed}")
                                st.session_state.causal_graph_rendered = True

                            # Header for causal inference
                        if st.session_state.get("causal_inference_ready", False):
                            st.divider()
                            st.header("♻️ 인과관계 추론 with Simple Rule", divider="rainbow")

                        # Graph visualization container
                        simple_rule_graph_container = st.container()

                        # Initialize random seed
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 111

                        # Function to extract directed edges with exclusions
                        def extract_directed_edges_with_exclusions(causal_graph, column_names, exclusions):
                            edges = []
                            num_nodes = len(causal_graph)

                            # Ensure graph and column_names align
                            if num_nodes != len(column_names):
                                raise ValueError(
                                    f"Mismatch between graph nodes ({num_nodes}) and column names ({len(column_names)})."
                                )

                            for i in range(num_nodes):
                                for j in range(num_nodes):
                                    if causal_graph[i, j] == 1 and causal_graph[j, i] != 1:  # Only i → j
                                        edge = (column_names[i], column_names[j])
                                        if edge not in exclusions:
                                            edges.append(edge)
                                    elif causal_graph[i, j] == -1 and causal_graph[j, i] != -1:  # Only j → i
                                        edge = (column_names[j], column_names[i])
                                        if edge not in exclusions:
                                            edges.append(edge)
                            return edges

                        # Run PC algorithm
                        cg = pc(X_transformed, alpha=0.05)

                        # Track transformed column names
                        transformed_columns = (
                            list(X_continuous.columns) +  # Continuous variables
                            list(preprocessor.named_transformers_['cat'].get_feature_names_out(X_categorical.columns))  # Categorical variables
                        )

                        # Validate graph size against column names
                        if len(transformed_columns) != len(cg.G.graph):
                            raise ValueError(
                                f"Mismatch between the number of graph nodes ({len(cg.G.graph)}) and transformed columns ({len(transformed_columns)})."
                            )

                        # Simple Rule-based graph rendering
                        with simple_rule_graph_container:
                            st.markdown("#### 🔔 Simple Rule 설정")
                            st.info("Simple Rule을 설정하여 특정 관계를 제외할 수 있습니다. (예: Age → Sex 또는 Sex → Age)")

                            # Simple Rule setup
                            available_columns = transformed_columns
                            exclude_edges = st.multiselect(
                                "✔️ 인과관계에서 제외할 관계를 선택해주세요:",
                                options=[f"{col1} → {col2}" for col1 in available_columns for col2 in available_columns if col1 != col2],
                                default=[]
                            )
                            excluded_edges = [(edge.split(" → ")[0], edge.split(" → ")[1]) for edge in exclude_edges]

                            # Extract filtered edges based on exclusions
                            filtered_edges = extract_directed_edges_with_exclusions(cg.G.graph, available_columns, excluded_edges)
                            filtered_causal_graph = nx.DiGraph()
                            filtered_causal_graph.add_edges_from(filtered_edges)

                            # Visualization function
                            def visualize_graph(graph, seed=None, padding_ratio=0.1, node_separation=1.5):
                                pos = nx.spring_layout(graph, seed=seed, k=node_separation, iterations=100)

                                edge_x = []
                                edge_y = []
                                annotations = []

                                for edge in graph.edges():
                                    x0, y0 = pos[edge[0]]
                                    x1, y1 = pos[edge[1]]
                                    dx, dy = x1 - x0, y1 - y0
                                    dist = (dx**2 + dy**2)**0.5

                                    # Apply padding
                                    x0_padded = x0 + dx * padding_ratio / dist
                                    y0_padded = y0 + dy * padding_ratio / dist
                                    x1_padded = x1 - dx * padding_ratio / dist
                                    y1_padded = y1 - dy * padding_ratio / dist

                                    edge_x.extend([x0_padded, x1_padded, None])
                                    edge_y.extend([y0_padded, y1_padded, None])

                                    annotations.append(
                                        dict(
                                            ax=x0_padded, ay=y0_padded,
                                            x=x1_padded, y=y1_padded,
                                            xref="x", yref="y",
                                            axref="x", ayref="y",
                                            showarrow=True,
                                            arrowhead=3,
                                            arrowsize=1.5,
                                            arrowwidth=1.5,
                                            arrowcolor="gray"
                                        )
                                    )

                                node_x = []
                                node_y = []
                                node_text = []
                                for node in graph.nodes():
                                    x, y = pos[node]
                                    node_x.append(x)
                                    node_y.append(y)
                                    node_text.append(node)

                                node_trace = go.Scatter(
                                    x=node_x,
                                    y=node_y,
                                    mode='markers+text',
                                    text=node_text,
                                    textfont=dict(family='Times New Roman', size=12, color='black'),
                                    marker=dict(
                                        size=60,
                                        color='lightblue',
                                        line=dict(width=2, color='darkblue')
                                    ),
                                    hoverinfo='text'
                                )

                                edge_trace = go.Scatter(
                                    x=edge_x,
                                    y=edge_y,
                                    line=dict(width=1.5, color='gray'),
                                    hoverinfo='none',
                                    mode='lines'
                                )

                                fig = go.Figure(data=[edge_trace, node_trace])
                                fig.update_layout(
                                    height=800,
                                    showlegend=False,
                                    title_text="Causal Graph (Filtered by Simple Rules)",
                                    title_font=dict(family="Times New Roman", size=20, color="black"),
                                    margin=dict(l=50, r=50, t=50, b=50),
                                    xaxis=dict(showgrid=False, zeroline=False),
                                    yaxis=dict(showgrid=False, zeroline=False),
                                    annotations=annotations
                                )

                                return fig

                            # Render the filtered causal graph
                            regenerate_layout_clicked = st.button("Simple Rule Figure 생성", key="regenerate_simple_rule_layout_button")

                            if regenerate_layout_clicked:
                                if len(excluded_edges) > 0:  # Check if any edges are excluded
                                    st.session_state.random_seed = np.random.randint(0, 9999)

                                    fig_simple_rule = visualize_graph(filtered_causal_graph, seed=st.session_state.random_seed)
                                    st.plotly_chart(fig_simple_rule, key=f"plotly_chart_simple_rule_{st.session_state.random_seed}")

                                    st.session_state["simple_rule_graph_rendered"] = True
                                else:
                                    st.error("❌ 관계를 하나 이상 선택해주세요.", icon="⚠️")

            # except ValueError as e:
            #     st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "💻 로지스틱 회귀분석":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">💻 로지스틱 회귀분석</h2>
            <p style="font-size:18px; color: #000000;">
            &nbsp;&nbsp;&nbsp;&nbsp;원하시는 데이터를 업로드하신 후 자유롭게 분석하세요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        st.warning("로지스틱 회귀는 범주형 타입의 종속변수를 분석합니다.", icon="🚨")
        uploaded_file = st.file_uploader("📁 로지스틱 회귀분석에 이용하실 데이터 파일을 업로드해주세요:")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="💻 로지스틱 회귀분석")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    st.divider()

                    # Variable selection section
                    st.markdown("<h4 style='color:grey;'>변수 선택</h4>", unsafe_allow_html=True)
                    st.markdown(
                    """
                    <style>
                    .custom-callout {
                        background-color: #f9f9f9;
                        padding: 10px;  /* Adjust padding */
                        border-radius: 10px;
                        border: 1px solid #d3d3d3;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }
                    .custom-callout p {
                        margin: 0;
                        color: #000000;  /* Font color */
                        font-size: 14px;  /* Font size */
                        line-height: 1.4; /* Line height */
                        text-align: left; /* Align text to the left */
                    }
                    </style>
                    <div class="custom-callout">
                        <p>- 로지스틱 회귀분석은 종속변수(y)가 범주형 변수일 경우 가능합니다.</p>
                        <p>- 범주형 설명변수는 unique 값이 5개 미만인 변수들로만 인식됩니다.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                    st.write(" ")

                    # Initialize session state for variables and processing states
                    if 'y_column' not in st.session_state:
                        st.session_state.y_column = None
                    if 'X_columns' not in st.session_state:
                        st.session_state.X_columns = []
                    if 'proceed_to_preprocessing' not in st.session_state:
                        st.session_state.proceed_to_preprocessing = False

                    y_column = st.selectbox(
                        "✔️ 종속변수(y)를 선택해주세요:",
                        options=["-- 선택 --"] + [col for col in df.columns if df[col].nunique() == 2],
                        index=0
                    )

                    def get_categorical_columns(df):
                        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
                        low_cardinality_numerical = [col for col in df.select_dtypes(exclude=['object', 'category']).columns if df[col].nunique() < 5]
                        return categorical_columns + low_cardinality_numerical

                    # 종속변수(y)가 선택된 경우에만 설명변수 선택 창 표시
                    if y_column != "-- 선택 --":
                        st.session_state.y_column = y_column

                        # Separate selections for continuous and categorical variables
                        continuous_columns = st.multiselect(
                            "✔️ 연속형 설명변수(X)를 선택해주세요:",
                            df.select_dtypes(include=['float64', 'int64']).columns,
                            key="continuous_columns_selection"
                        )

                        categorical_columns = st.multiselect(
                            "✔️ 범주형 설명변수(X)를 선택해주세요:",
                            get_categorical_columns(df),
                            key="categorical_columns_selection"
                        )

                        st.session_state.X_columns = continuous_columns + categorical_columns

                        # Add a button to confirm the selections
                        if st.button('🚀 선택 완료', key='complete_button'):
                            if y_column and (continuous_columns or categorical_columns):  # Ensure that y and at least one X is selected
                                st.session_state.y_column = y_column
                                st.session_state.continuous_columns = continuous_columns
                                st.session_state.categorical_columns = categorical_columns
                                st.session_state.proceed_to_preprocessing = True
                            else:
                                st.warning("종속변수와 설명변수를 한 개 이상 선택해주세요.", icon="⚠️")

                            # Drop rows with missing values in the dependent variable (y)
                            df = df.dropna(subset=[st.session_state.y_column])

                            # Separate continuous and categorical columns
                            X_continuous = df[st.session_state.continuous_columns]
                            X_categorical = df[st.session_state.categorical_columns]
                            y = df[st.session_state.y_column]

                        # 전처리 단계 (선택 완료 후 진행)
                        if st.session_state.get("proceed_to_preprocessing", False):
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>결측 파악 및 처리</h4>", unsafe_allow_html=True)

                            # Check missing values in y
                            y_missing_count = df[st.session_state.y_column].isna().sum()
                            if y_missing_count > 0:
                                st.warning(f"선택된 종속변수 '{st.session_state.y_column}'에 결측값이 {y_missing_count}개 있습니다. 해당 행은 분석에서 제외됩니다.", icon="⚠️")
                                df = df.dropna(subset=[st.session_state.y_column])

                            # 데이터 분리
                            X_continuous = df[st.session_state.continuous_columns]
                            X_categorical = df[st.session_state.categorical_columns]
                            y = df[st.session_state.y_column]

                            # Check for missing values in independent variables
                            continuous_missing = X_continuous.isnull().any().any()
                            categorical_missing = X_categorical.isnull().any().any()

                            if not continuous_missing and not categorical_missing:
                                st.success("결측 처리 작업 없이 분석이 가능합니다.", icon="✅")
                                st.session_state.logit_ready = True
                            else:
                                # Missing value handling
                                continuous_missing_value_strategies = {}
                                categorical_missing_value_strategies = {}

                                # Continuous variables
                                for column in st.session_state.continuous_columns:
                                    missing_count = df[column].isna().sum()
                                    if missing_count > 0:
                                        st.error(f"선택하신 범주형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                        strategy = st.selectbox(
                                            f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                            key=f"continuous_{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            continuous_missing_value_strategies[column] = strategy

                                # Categorical variables
                                for column in st.session_state.categorical_columns:
                                    missing_count = df[column].isna().sum()
                                    if missing_count > 0:
                                        st.error(f"선택하신 범주형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                        strategy = st.selectbox(
                                            f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                            key=f"categorical_{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            categorical_missing_value_strategies[column] = strategy

                                # Apply missing value handling and add "분석 시작" button
                                if st.button("🚀 분석 시작"):
                                    # Process continuous variables
                                    for column, strategy in continuous_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_continuous = X_continuous.dropna(subset=[column])
                                        else:
                                            impute_strategy = {
                                                '해당 열의 평균값으로 대체': 'mean',
                                                '해당 열의 중앙값으로 대체': 'median',
                                                '해당 열의 최빈값으로 대체': 'most_frequent'
                                            }[strategy]
                                            imputer = SimpleImputer(strategy=impute_strategy)
                                            X_continuous[[column]] = imputer.fit_transform(X_continuous[[column]])

                                    # Process categorical variables
                                    for column, strategy in categorical_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_categorical = X_categorical.dropna(subset=[column])
                                        elif strategy == '해당 열의 최빈값으로 대체':
                                            imputer = SimpleImputer(strategy='most_frequent')
                                            X_categorical[[column]] = imputer.fit_transform(X_categorical[[column]])

                                    # Synchronize indexes
                                    shared_indexes = X_continuous.index.intersection(X_categorical.index)
                                    X_continuous = X_continuous.loc[shared_indexes]
                                    X_categorical = X_categorical.loc[shared_indexes]
                                    y = y.loc[shared_indexes]

                                    # 갱신된 데이터 상태를 session_state에 저장
                                    st.session_state.X_continuous = X_continuous
                                    st.session_state.X_categorical = X_categorical
                                    st.session_state.y = y
                                    st.session_state.logit_ready = True  # Flag to indicate that data is ready for modeling

                                    # 디버깅용 출력
                                    st.success("결측값 처리가 완료되었습니다. 분석을 진행하세요.", icon="✅")

                                # 분석 시작 버튼이 눌린 경우
                                if st.session_state.get("logit_ready", False):
                                    st.divider()
                                    st.header('💻 로지스틱 회귀분석 결과', divider='rainbow')

                                    # 갱신된 데이터를 session_state에서 불러오기
                                    X_continuous = st.session_state.X_continuous
                                    X_categorical = st.session_state.X_categorical
                                    y = st.session_state.y

                                    # One-Hot Encoding for categorical variables
                                    X_categorical = pd.get_dummies(X_categorical, drop_first=True)

                                    # Boolean 처리
                                    for column in X_categorical.columns:
                                        if X_categorical[column].dtype == 'bool':
                                            X_categorical[column] = X_categorical[column].map({True: 1, False: 0})

                                    # Combine continuous and categorical variables
                                    X = pd.concat([X_continuous, X_categorical], axis=1)

                                    # Ensure all columns in X are numeric
                                    X = X.apply(pd.to_numeric, errors='coerce')

                                    # 결측 및 무한 값 확인
                                    if X.isnull().values.any():
                                        st.error("전처리 후에도 설명변수에 결측치가 남아 있습니다. 결측치 처리를 확인해주세요.")
                                        st.dataframe(X)  # 디버깅용 데이터 출력
                                    elif np.isinf(X).values.any():
                                        st.error("전처리 후 설명변수에 무한 값이 존재합니다. 데이터 정규화를 확인해주세요.")
                                        st.dataframe(X)  # 디버깅용 데이터 출력
                                    else:
                                        try:
                                            # Split the data
                                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                                            # Add constant (intercept) to the features
                                            X_train_const = sm.add_constant(X_train)
                                            X_test_const = sm.add_constant(X_test)

                                            # Logistic regression model using statsmodels
                                            model = sm.Logit(y_train, X_train_const)
                                            result = model.fit()

                                            # Predictions
                                            y_pred_prob = result.predict(X_test_const)
                                            y_pred_class = (y_pred_prob >= 0.5).astype(int)

                                            # 결과 출력
                                            st.markdown("<h4 style='font-size:16px;'>Model OR & P-value:</h4>", unsafe_allow_html=True)
                                            summary_table = result.summary2().tables[1]
                                            summary_df = summary_table[['Coef.', 'P>|z|', '[0.025', '0.975]']]

                                            # Calculate Odds Ratio (OR) as the exponential of the coefficient
                                            summary_df['OR'] = np.exp(summary_df['Coef.'])
                                            summary_df['95% CI Lower'] = np.exp(summary_df['[0.025'])
                                            summary_df['95% CI Upper'] = np.exp(summary_df['0.975]'])

                                            # Rename columns for clarity
                                            summary_df = summary_df.rename(columns={'P>|z|': 'P-value'})

                                            # Replace P-values equal to 0 with "<0.001"
                                            summary_df['P-value'] = summary_df['P-value'].apply(lambda x: '<0.001' if x == 0 else round(x, 4))

                                            # Rearrange columns to include OR, Coefficient, P-value, and Confidence Interval
                                            summary_df = summary_df[['OR', '95% CI Lower', '95% CI Upper', 'P-value']]
                                            st.dataframe(summary_df, use_container_width=True)

                                            # Classification Report
                                            st.markdown("<h5 style='font-size:16px;'><strong>Classification Report:</strong></h5>", unsafe_allow_html=True)
                                            report = classification_report(y_test, y_pred_class, output_dict=True)
                                            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                                            st.write(" ")
                                            st.write(" ")
                                            st.header("💻 분석결과 Figure", divider='rainbow')
                                            # AUC Curve
                                            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                                            roc_auc = auc(fpr, tpr)

                                            # Create a Plotly figure for ROC Curve
                                            fig_roc = go.Figure()

                                            # Add ROC curve line
                                            fig_roc.add_trace(go.Scatter(
                                                x=fpr, y=tpr,
                                                mode='lines',
                                                name=f'ROC curve (area = {roc_auc:.2f})',
                                                line=dict(color='darkorange', width=2)
                                            ))

                                            # Add diagonal line
                                            fig_roc.add_trace(go.Scatter(
                                                x=[0, 1], y=[0, 1],
                                                mode='lines',
                                                line=dict(color='navy', width=2, dash='dash'),
                                                showlegend=False
                                            ))

                                            # Update layout for Plotly figure
                                            fig_roc.update_layout(
                                                title="ROC Curve",
                                                xaxis_title="False Positive Rate",
                                                yaxis_title="True Positive Rate",
                                                legend=dict(x=0.4, y=0),
                                                width=600, height=600  # Adjust dimensions as per your requirement
                                            )

                                            # Display the Plotly figure in Streamlit
                                            st.plotly_chart(fig_roc)

                                            # Create confusion matrix
                                            cm = confusion_matrix(y_test, y_pred_class)

                                            # Create a Plotly heatmap for confusion matrix
                                            fig_cm = ff.create_annotated_heatmap(
                                                z=cm,
                                                x=['Predicted Negative', 'Predicted Positive'],
                                                y=['Actual Negative', 'Actual Positive'],
                                                colorscale='Blues',
                                                showscale=False,
                                                annotation_text=[[str(value) for value in row] for row in cm]  # Add annotations with values
                                            )

                                            # Update annotations to change font size
                                            for annotation in fig_cm.layout.annotations:
                                                annotation.font.size = 16  # Adjust font size
                                                annotation.font.color = "black"  # Change font color for better contrast

                                            # Update layout for the confusion matrix
                                            fig_cm.update_layout(
                                                title="Confusion Matrix",
                                                width=600, height=600  # Adjust dimensions as per your requirement
                                            )

                                            # Display the Plotly heatmap in Streamlit
                                            st.plotly_chart(fig_cm)

                                            # Filter out "const" variable from the summary_df
                                            summary_df = summary_df[~summary_df.index.str.contains('const')]

                                            # Drop rows with NaN in CI or OR columns
                                            summary_df = summary_df.dropna(subset=['95% CI Lower', '95% CI Upper', 'OR'])

                                            # Calculate X-axis range (ensure CI fits and log scale works)
                                            x_min = 0  # Set minimum to avoid log(0)
                                            x_max = summary_df['95% CI Upper'].max()+1

                                            # Apply log transformation safely
                                            log_x_min = np.log10(x_min)
                                            log_x_max = np.log10(x_max)

                                            # Initialize the figure
                                            fig_forest = go.Figure()

                                            # Add horizontal lines for confidence intervals
                                            for i, row in summary_df.iterrows():
                                                # Add CI line
                                                fig_forest.add_trace(go.Scatter(
                                                    x=[row['95% CI Lower'], row['95% CI Upper']],
                                                    y=[i, i],
                                                    mode='lines',
                                                    line=dict(color='gray', width=2),
                                                    showlegend=False
                                                ))

                                                # Add OR point
                                                fig_forest.add_trace(go.Scatter(
                                                    x=[row['OR']],
                                                    y=[i],
                                                    mode='markers',
                                                    marker=dict(color='blue', size=7),
                                                    showlegend=False
                                                ))

                                                # Add CI end markers ("|") with thicker appearance
                                                fig_forest.add_trace(go.Scatter(
                                                    x=[row['95% CI Lower'], row['95% CI Upper']],
                                                    y=[i, i],
                                                    mode='text',
                                                    text=["|", "|"],
                                                    textfont=dict(size=18, color="gray", family="Arial Black"),  # Bold and larger font
                                                    textposition="middle center",
                                                    showlegend=False
                                                ))

                                            # Add vertical line for OR=1
                                            fig_forest.add_shape(
                                                type="line",
                                                x0=1, x1=1,
                                                y0=-0.5, y1=len(summary_df) - 0.5,
                                                line=dict(color="red", width=2, dash="dash")
                                            )

                                            # Update layout for the forest plot
                                            fig_forest.update_layout(
                                                title="Forest Plot of Odds Ratios",
                                                xaxis=dict(
                                                    title="Odds Ratio",
                                                    type="log",  # Log scale for better visualization
                                                    range=[np.log10(x_min), np.log10(x_max)],
                                                    zeroline=False
                                                ),
                                                yaxis=dict(
                                                    title="Variables",
                                                    tickvals=list(range(len(summary_df))),
                                                    ticktext=summary_df.index,  # Use index names (variables) as y-axis labels
                                                    autorange="reversed"  # Reverse Y-axis to match conventional forest plot style
                                                ),
                                                width=800,
                                                height=600,
                                                template="plotly_white"
                                            )

                                            # Display the Plotly figure in Streamlit
                                            st.plotly_chart(fig_forest)

                                        except Exception as e:
                                            st.error(f"모델 학습 중 오류가 발생했습니다.")
                                            st.error("자세한 오류 정보: ", traceback.format_exc())  # 스택 트레이스 출력

            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "💻 생존분석":
    # Streamlit setup
        st.markdown(
            """
            <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
                <h2 style="color: #000000;">💻 생존분석</h2>
                <p style="font-size:18px; color: #000000;">
                &nbsp;&nbsp;&nbsp;&nbsp;원하시는 데이터를 업로드하신 후 자유롭게 분석하세요.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # 1. 파일 업로드
        st.markdown("<h4 style='color:grey;'>데이터 업로드</h4>", unsafe_allow_html=True)
        st.warning("생존분석은 생존 시간과 상태(생존/사망 등)를 포함하는 데이터를 필요로 합니다.", icon="🚨")
        uploaded_file = st.file_uploader("📁 생존분석에 이용하실 데이터 파일을 업로드해주세요:")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    st.error("❌ 파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")
                    st.stop()

                # 파일 이름 해싱 및 임시 저장
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 파일 이름을 해시로 익명화
                    file_hash = hashlib.sha256(uploaded_file.name.encode()).hexdigest()
                    temp_file_path = os.path.join(temp_dir, file_hash)

                    # 업로드된 파일을 임시 디렉터리에 저장
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getbuffer())

                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(temp_file_path)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(temp_file_path)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션
                        if len(sheet_names) > 1:
                            sheet = st.selectbox("✔️ 업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요:", 
                                                ["-- 선택 --"] + sheet_names)
                            if sheet == "-- 선택 --":
                                st.stop()
                            else:
                                df = pd.read_excel(temp_file_path, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            df = pd.read_excel(temp_file_path, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("✔️ 옵션을 선택해주세요:", ["데이터", "결측수", "요약통계"], key="💻 생존분석")

                    # Show the corresponding output based on the user's selection
                    if selected_option == "데이터":
                        # Display the data
                        st.dataframe(df, use_container_width=True)

                    elif selected_option == "결측수":
                        # Calculate counts, missing counts, and percentages
                        counts = df.notna().sum()
                        missing_counts = df.isna().sum()
                        missing_percentages = (df.isna().mean()) * 100

                        # Format counts and missing counts with 1000 separators and percentages
                        counts_formatted = counts.apply(lambda x: f"{x:,}")
                        missing_counts_formatted = missing_counts.apply(lambda x: f"{x:,}")
                        missing_percentages_formatted = missing_percentages.round(2).astype(str) + '%'

                        # Create a DataFrame with the formatted information
                        missing_info = pd.DataFrame({
                            'Columns': df.columns,
                            'Count': counts_formatted,
                            'Missing Count': missing_counts_formatted,
                            'Missing Percentage': missing_percentages_formatted
                        }).reset_index(drop=True)

                        # Display the missing information
                        st.dataframe(missing_info, use_container_width=True)

                    elif selected_option == "요약통계":
                        # Display summary statistics
                        st.dataframe(df.describe(), use_container_width=True)

                    st.divider()

                    # Section for variable selection
                    duration_column = None  # Initialize duration_column as None
                    
                    # Session state 초기화
                    if 'analysis_ready' not in st.session_state:
                        st.session_state.analysis_ready = False

                    # Step 1: 변수 선택
                    st.markdown("<h4 style='color:grey;'>☑️ 변수 선택</h4>", unsafe_allow_html=True)
                    use_duration_column = st.checkbox("생존 '기간' 열이 이미 존재합니까?", key="use_duration_column")

                    if use_duration_column:
                        # 생존 기간 열이 존재할 경우
                        duration_column = st.selectbox("✔️ 생존 기간을 나타내는 열을 선택해주세요:", options=["-- 선택 --"] + list(df.columns), index=0, key="duration_column")
                        event_column = st.selectbox("✔️ 생존 상태(1=이벤트 발생, 0=검열)를 나타내는 열을 선택해주세요:", options=["-- 선택 --"] + list(df.columns), index=0, key="event_column")
                    else:
                        # 생존 기간 열이 없을 경우
                        time_column = st.selectbox("✔️ 생존(검열)일자를 나타내는 열을 선택해주세요:", options=["-- 선택 --"] + list(df.columns), index=0, key="time_column")
                        event_column = st.selectbox("✔️ 생존 상태(1=이벤트 발생, 0=검열)를 나타내는 열을 선택해주세요:", options=["-- 선택 --"] + list(df.columns), index=0, key="event_column_alt")

                    # Step 2: 기준 날짜 설정 (생존 기간 열이 없을 때만)
                    if not use_duration_column:
                        if time_column != "-- 선택 --" and event_column != "-- 선택 --":
                            st.markdown("<h4 style='color:grey;'>☑️ 기준 날짜 설정</h4>", unsafe_allow_html=True)
                            baseline_date = st.date_input("기준 날짜를 입력해주세요 (YYYY-MM-DD):", key="baseline_date")

                            # time_column 변환
                            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

                            # Step 3: 결측 파악 및 처리 (선택된 열이 유효한 경우에만)
                            st.markdown("<h4 style='color:grey;'>☑️ 결측 파악 및 처리</h4>", unsafe_allow_html=True)

                            missing_time_count = df[time_column].isna().sum()
                            missing_event_count = df[event_column].isna().sum()

                            # 결측 메시지 출력
                            if missing_time_count > 0 or missing_event_count > 0:
                                st.markdown(
                                    f"<p style='font-size:16px; color:red;'>"
                                    f"<strong>'{time_column}' 열에 {missing_time_count}개의 결측이, "
                                    f"'{event_column}' 열에 {missing_event_count}개의 결측이 발견되었습니다.</strong></p>",
                                    unsafe_allow_html=True
                                )
                                if st.checkbox("✔️ 결측된 관측을 검열로 기록하시겠습니까?", key="handle_missing"):
                                    censoring_date = st.date_input(
                                        f"'{time_column}' 열의 결측값을 채우기 위해 검열일자를 입력해주세요 (YYYY-MM-DD):",
                                        key="censoring_date"
                                    )
                                    if censoring_date:
                                        censoring_date = pd.to_datetime(censoring_date)
                                        df[time_column] = df[time_column].fillna(censoring_date)
                                        df[event_column] = df[event_column].fillna(0)  # 검열로 처리
                            else:
                                st.success("결측 처리 작업 없이 분석이 가능합니다.")

                    # Step 4: Duration 계산 (결측 처리 이후)
                    if not use_duration_column and time_column != "-- 선택 --" and event_column != "-- 선택 --" and "baseline_date" in locals():
                        if df[time_column].isna().sum() == 0:  # 결측값이 없는 경우에만 계산
                            duration_column = "duration"
                            df[duration_column] = (df[time_column] - pd.to_datetime(baseline_date)).dt.days

                    # 분석 시작 버튼
                    st.write(" ")
                    if st.button("🚀 분석 시작", key="start_analysis"):
                        st.session_state.analysis_ready = True

                    # 분석 결과 표시
                    if st.session_state.analysis_ready:
                        st.divider()
                        st.header("💻 생존분석 결과", divider='rainbow')

                        if use_duration_column:
                            # 기간 열이 존재하는 경우
                            df_to_display = df[[event_column, duration_column]]
                            durations = df[duration_column].dropna()
                        else:
                            # 기간 열이 없는 경우
                            df["baseline_date"] = baseline_date  # 기준 날짜 추가
                            df_to_display = df[[event_column, "baseline_date", time_column, duration_column]]
                            durations = df[duration_column].dropna()

                        events = df[event_column].dropna()

                        # 데이터프레임 표시
                        st.markdown("<h6>Event Dataframe</h6>", unsafe_allow_html=True)
                        st.dataframe(df_to_display, use_container_width=True)

                        # Kaplan-Meier 분석
                        st.markdown("<h6>Event Table</h6>", unsafe_allow_html=True)
                        from lifelines import KaplanMeierFitter
                        kmf = KaplanMeierFitter()
                        kmf.fit(durations, event_observed=events)
                        event_table = kmf.event_table
                        st.dataframe(event_table, use_container_width=True)

                        st.write(" ")
                        st.header("💻 분석결과 Figure", divider='rainbow')

                        # Kaplan-Meier 생존 곡선
                        plt.figure(figsize=(10, 6))
                        kmf.plot_survival_function()

                        # Plot Kaplan-Meier curve using Plotly
                        km_curve = go.Figure()
                        km_curve.add_trace(go.Scatter(
                            x=kmf.timeline,
                            y=kmf.survival_function_['KM_estimate'],
                            mode='lines',
                            name='Kaplan-Meier Curve',
                            line=dict(color='blue')
                        ))
                        km_curve.update_layout(
                            title="Kaplan-Meier Curve",
                            xaxis_title="Duration (days)",
                            yaxis_title="Survival Probability",
                            width=800,
                            height=600
                        )
                        st.plotly_chart(km_curve)

                        # Divider for categorical analysis
                        st.divider()
                        st.markdown("<h6>Kaplan-Meier Curve with Variables</h6>", unsafe_allow_html=True)

                        # Categorical variable selection for Kaplan-Meier curve
                        excluded_columns = ["baseline_date", duration_column, event_column]

                        if not use_duration_column:
                            excluded_columns.append(time_column)

                        km_cat_column = st.selectbox(
                            "✔️ KM Curve를 볼 변수를 선택해주세요:",
                            options=["-- 선택 --"] + [col for col in df.columns if col not in excluded_columns and df[col].nunique() < 10],
                            index=0,
                            key="km_cat_column"
                        )

                        # Display the selected DataFrame and grouped KM curves if a variable is selected
                        if st.session_state.km_cat_column != "-- 선택 --":
                            st.markdown("<h6>Event Dataframe</h6>", unsafe_allow_html=True)

                            try:
                                # DataFrame display logic
                                if use_duration_column:
                                    # 생존 기간 열 사용 시
                                    st.dataframe(
                                        df[[event_column, duration_column, st.session_state.km_cat_column]], use_container_width=True
                                    )
                                else:
                                    # 생존(검열)일자 열 사용 시
                                    st.dataframe(
                                        df[[event_column, time_column, duration_column, st.session_state.km_cat_column]], use_container_width=True
                                    )
                            except KeyError as e:
                                st.error(f"선택된 열 중 하나가 데이터프레임에 없습니다. 오류: {e}")

                            # Plot Kaplan-Meier curves grouped by the categorical variable
                            km_curve = go.Figure()

                            for group in df[st.session_state.km_cat_column].dropna().unique():
                                # Filter the DataFrame for the current group
                                group_df = df[df[st.session_state.km_cat_column] == group]

                                # Extract durations and events
                                group_durations = group_df[duration_column].dropna()
                                group_events = group_df[event_column].dropna()

                                # Ensure non-empty groups
                                if not group_durations.empty and not group_events.empty:
                                    kmf.fit(group_durations, event_observed=group_events, label=str(group))

                                    # Add trace
                                    km_curve.add_trace(go.Scatter(
                                        x=kmf.timeline,
                                        y=kmf.survival_function_[kmf.survival_function_.columns[0]],
                                        mode='lines',
                                        name=f'{st.session_state.km_cat_column}: {group}'
                                    ))

                            # Update layout for Plotly figure
                            km_curve.update_layout(
                                title="Kaplan-Meier Curve by Category",
                                xaxis_title="Duration (days)",
                                yaxis_title="Survival Probability",
                                width=800,
                                height=600
                            )
                            st.plotly_chart(km_curve)

                            for group in df[st.session_state.km_cat_column].dropna().unique():  # Ensure no NaN groups
                                # Display the event table for the current group
                                st.markdown(f"<h6>Event Table for {st.session_state.km_cat_column}: {group}</h6>", unsafe_allow_html=True)
                                event_table = kmf.event_table
                                st.dataframe(event_table, use_container_width=True)

                        # Style for custom callout boxes
                        st.divider()
                        st.markdown(
                            """
                            <style>
                            .custom-callout {
                                background-color: #f9f9f9;
                                padding: 10px;
                                border-radius: 10px;
                                border: 1px solid #d3d3d3;
                                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                            }
                            .custom-callout p {
                                margin: 0;
                                color: #000000;
                                font-size: 14px;
                                line-height: 1.4;
                                text-align: left;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                        # st.session_state 초기화
                        if "continuous_columns" not in st.session_state:
                            st.session_state.continuous_columns = []
                        if "categorical_columns" not in st.session_state:
                            st.session_state.categorical_columns = []
                        if "proceed_to_preprocessing" not in st.session_state:
                            st.session_state.proceed_to_preprocessing = False
                        if "survival_ready" not in st.session_state:
                            st.session_state.survival_ready = False

                        # 범주형 변수 추출 함수
                        def get_categorical_columns(df):
                            categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
                            low_cardinality_numerical = [
                                col for col in df.select_dtypes(exclude=['object', 'category']).columns if df[col].nunique() < 5
                            ]
                            return categorical_columns + low_cardinality_numerical

                        # UI for variable selection
                        st.header("💻 Cox Proportional Hazard Model", divider='rainbow')
                        st.markdown("<h4 style='color:grey;'>변수 선택</h4>", unsafe_allow_html=True)

                        # 연속형 변수 선택 (제외된 열 제외)
                        continuous_columns = st.multiselect(
                            "✔️ 연속형 변수를 선택해주세요:",
                            [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in excluded_columns],
                            key="continuous_columns_selection"
                        )

                        # 범주형 변수 선택 (제외된 열 제외)
                        categorical_columns = st.multiselect(
                            "✔️ 범주형 변수를 선택해주세요:",
                            [col for col in get_categorical_columns(df) if col not in excluded_columns],
                            key="categorical_columns_selection"
                        )

                        # 선택 완료 버튼
                        if st.button('🚀 선택 완료', key='complete_button'):
                            if len(continuous_columns) + len(categorical_columns) > 1:
                                st.session_state.continuous_columns = continuous_columns
                                st.session_state.categorical_columns = categorical_columns
                                st.session_state.proceed_to_preprocessing = True
                                st.success("변수 선택이 완료되었습니다. 다음 단계로 진행하세요.", icon="✅")
                            else:
                                st.warning("변수를 두 개 이상 선택하셔야 합니다.", icon="⚠️")

                        # 2. 전처리 단계
                        if st.session_state.get("proceed_to_preprocessing", False):
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                            # 데이터 분리
                            X_continuous = df[st.session_state.continuous_columns]
                            X_categorical = df[st.session_state.categorical_columns]

                            # 결측값 유무 확인
                            continuous_missing = X_continuous.isnull().any().any()
                            categorical_missing = X_categorical.isnull().any().any()

                            if not continuous_missing and not categorical_missing:
                                st.success("결측 처리 작업 없이 분석이 가능합니다.", icon="✅")
                                st.session_state.survival_ready = True  # 바로 분석 준비 완료
                            else:
                                st.warning("결측값 처리가 필요합니다.", icon="⚠️")
                                
                                # 결측 처리 로직
                                continuous_missing_value_strategies = {}
                                categorical_missing_value_strategies = {}

                                # Step 2: 연속형 변수 결측 처리 선택
                                for column in st.session_state.continuous_columns:
                                    missing_count = df[column].isna().sum()
                                    if missing_count > 0:
                                        st.error(f"선택하신 연속형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                        strategy = st.selectbox(
                                            f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                            key=f"continuous_{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            continuous_missing_value_strategies[column] = strategy

                                # Step 3: 범주형 변수 결측 처리 선택
                                for column in st.session_state.categorical_columns:
                                    missing_count = df[column].isna().sum()
                                    if missing_count > 0:
                                        st.error(f"선택하신 범주형 변수 '{column}'에 결측치 {missing_count}개가 있습니다.", icon="⛔")
                                        strategy = st.selectbox(
                                            f"✔️ '{column}'의 결측 처리 방법을 선택해주세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                            key=f"categorical_{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            categorical_missing_value_strategies[column] = strategy

                                # Step 4: 결측 처리 버튼
                                if st.button("결측 처리"):
                                    # 연속형 변수 결측 처리
                                    for column, strategy in continuous_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_continuous = X_continuous.dropna(subset=[column])
                                        else:
                                            impute_strategy = {
                                                '해당 열의 평균값으로 대체': 'mean',
                                                '해당 열의 중앙값으로 대체': 'median',
                                                '해당 열의 최빈값으로 대체': 'most_frequent'
                                            }[strategy]
                                            imputer = SimpleImputer(strategy=impute_strategy)
                                            X_continuous[[column]] = imputer.fit_transform(X_continuous[[column]])

                                    # 범주형 변수 결측 처리
                                    for column, strategy in categorical_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_categorical = X_categorical.dropna(subset=[column])
                                        elif strategy == '해당 열의 최빈값으로 대체':
                                            imputer = SimpleImputer(strategy='most_frequent')
                                            X_categorical[[column]] = imputer.fit_transform(X_categorical[[column]])

                                    # Step 5: 인덱스 동기화
                                    shared_indexes = X_continuous.index.intersection(X_categorical.index)
                                    X_continuous = X_continuous.loc[shared_indexes]
                                    X_categorical = X_categorical.loc[shared_indexes]

                                    st.success("결측값 처리가 완료되었습니다. 분석을 진행하세요.", icon="✅")
                                    st.session_state.survival_ready = True

                            # Step 6: 모델 학습 시작 버튼
                            if st.session_state.get("survival_ready", False) and st.button('🚀 모델 학습 시작', key='train_model_button'):
                                st.divider()

                                # Fit the Cox Proportional Hazards Model
                                try:
                                    # Handle categorical variables
                                    X_categorical = pd.get_dummies(X_categorical, drop_first=True)

                                    # Check and handle boolean columns if they exist
                                    for column in X_categorical.columns:
                                        if X_categorical[column].dtype == 'bool':  # Only map if the column is of boolean type
                                            X_categorical[column] = X_categorical[column].map({True: 1, False: 0})

                                    # Ensure all columns in X_categorical are integers
                                    X_categorical = X_categorical.astype(int, errors='ignore')

                                    # Combine continuous and categorical columns
                                    X = pd.concat([X_continuous, X_categorical], axis=1)

                                    # Ensure that combined X does not have object or mixed types
                                    X = X.apply(pd.to_numeric, errors='coerce')

                                    # Combine X with duration and event columns
                                    df_for_cox = pd.concat([df[[duration_column, event_column]], X], axis=1).dropna()

                                    train_data, test_data = train_test_split(
                                        df_for_cox, 
                                        test_size=0.2, 
                                        random_state=42, 
                                        stratify=df_for_cox[event_column]
                                    )

                                    # Fit the Cox Proportional Hazards Model on training data
                                    cph = CoxPHFitter()
                                    cph.fit(train_data, duration_col=duration_column, event_col=event_column)

                                    # Display Cox model summary
                                    summary_table = cph.summary
                                    summary_df = summary_table[['coef', 'p', 'coef lower 95%', 'coef upper 95%']]
                                    summary_df['HR'] = np.exp(summary_df['coef'])
                                    summary_df['95% CI Lower'] = np.exp(summary_df['coef lower 95%'])
                                    summary_df['95% CI Upper'] = np.exp(summary_df['coef upper 95%'])

                                    # Display results
                                    st.header("💻 Cox Proportional Hazards Model 결과", divider="rainbow")

                                    # Evaluate model on test data
                                    c_index = cph.score(test_data)
                                    st.write(f"**Concordance Index (Test Data): {c_index:.3f}**")

                                    # Rename columns for clarity
                                    summary_df = summary_df.rename(columns={'p': 'P-value'})
                                    st.dataframe(summary_df[['HR', '95% CI Lower', '95% CI Upper', 'P-value']], use_container_width=True)

                                    # Predict survival probabilities for test data
                                    survival_probabilities = cph.predict_survival_function(test_data)

                                    # Convert survival probabilities to event predictions (threshold: 0.5)
                                    predicted_probs = 1 - survival_probabilities.iloc[-1, :]  # Probability of the event occurring at the last time point
                                    predicted_probs.index = test_data.index  # Align indices
                                    y_pred_class = (predicted_probs >= 0.5).astype(int)  # Classify based on threshold

                                    # Ensure y_test is aligned
                                    y_test = test_data[event_column]  # Extract true event labels from test data

                                    # Classification Report
                                    st.markdown("<h5 style='font-size:16px;'><strong>Classification Report:</strong></h5>", unsafe_allow_html=True)
                                    report = classification_report(y_test, y_pred_class, output_dict=True)
                                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                                    st.write(" ")
                                    st.write(" ")
                                    st.header("💻 분석결과 Figure", divider='rainbow')

                                    # ROC Curve
                                    fpr, tpr, _ = roc_curve(y_test, predicted_probs)
                                    roc_auc = auc(fpr, tpr)

                                    # Create a Plotly figure for ROC Curve
                                    fig_roc = go.Figure()

                                    # Add ROC curve line
                                    fig_roc.add_trace(go.Scatter(
                                        x=fpr, y=tpr,
                                        mode='lines',
                                        name=f'ROC curve (area = {roc_auc:.2f})',
                                        line=dict(color='darkorange', width=2)
                                    ))

                                    # Add diagonal line
                                    fig_roc.add_trace(go.Scatter(
                                        x=[0, 1], y=[0, 1],
                                        mode='lines',
                                        line=dict(color='navy', width=2, dash='dash'),
                                        showlegend=False
                                    ))

                                    # Update layout for Plotly figure
                                    fig_roc.update_layout(
                                        title="ROC Curve",
                                        xaxis_title="False Positive Rate",
                                        yaxis_title="True Positive Rate",
                                        legend=dict(x=0.4, y=0),
                                        width=600, height=600  # Adjust dimensions as per your requirement
                                    )

                                    # Display the Plotly figure in Streamlit
                                    st.plotly_chart(fig_roc)

                                    # Confusion Matrix
                                    cm = confusion_matrix(y_test, y_pred_class)

                                    # Create a Plotly heatmap for confusion matrix
                                    fig_cm = ff.create_annotated_heatmap(
                                        z=cm,
                                        x=['Predicted Negative', 'Predicted Positive'],
                                        y=['Actual Negative', 'Actual Positive'],
                                        colorscale='Blues',
                                        showscale=False,
                                        annotation_text=[[str(value) for value in row] for row in cm]  # Add annotations with values
                                    )

                                    # Update annotations to change font size
                                    for annotation in fig_cm.layout.annotations:
                                        annotation.font.size = 16  # Adjust font size
                                        annotation.font.color = "black"  # Change font color for better contrast

                                    # Update layout for the confusion matrix
                                    fig_cm.update_layout(
                                        title="Confusion Matrix",
                                        width=600, height=600  # Adjust dimensions as per your requirement
                                    )

                                    # Display the Plotly heatmap in Streamlit
                                    st.plotly_chart(fig_cm)

                                    # Filter out "const" variable from the summary_df
                                    summary_df = summary_df[~summary_df.index.str.contains('const')]

                                    # Drop rows with NaN in CI or OR columns
                                    summary_df = summary_df.dropna(subset=['95% CI Lower', '95% CI Upper', 'HR'])

                                    # Calculate X-axis range (ensure CI fits and log scale works)
                                    x_min = 0  # Set minimum to avoid log(0)
                                    x_max = summary_df['95% CI Upper'].max()+1

                                    # Apply log transformation safely
                                    log_x_min = np.log10(x_min)
                                    log_x_max = np.log10(x_max)

                                    # Initialize the figure
                                    fig_forest = go.Figure()

                                    # Add horizontal lines for confidence intervals
                                    for i, row in summary_df.iterrows():
                                        # Add CI line
                                        fig_forest.add_trace(go.Scatter(
                                            x=[row['95% CI Lower'], row['95% CI Upper']],
                                            y=[i, i],
                                            mode='lines',
                                            line=dict(color='gray', width=2),
                                            showlegend=False
                                        ))

                                        # Add HR point
                                        fig_forest.add_trace(go.Scatter(
                                            x=[row['HR']],
                                            y=[i],
                                            mode='markers',
                                            marker=dict(color='blue', size=7),
                                            showlegend=False
                                        ))

                                        # Add CI end markers ("|") with thicker appearance
                                        fig_forest.add_trace(go.Scatter(
                                            x=[row['95% CI Lower'], row['95% CI Upper']],
                                            y=[i, i],
                                            mode='text',
                                            text=["|", "|"],
                                            textfont=dict(size=18, color="gray", family="Arial Black"),  # Bold and larger font
                                            textposition="middle center",
                                            showlegend=False
                                        ))

                                    # Add vertical line for HR=1
                                    fig_forest.add_shape(
                                        type="line",
                                        x0=1, x1=1,
                                        y0=-0.5, y1=len(summary_df) - 0.5,
                                        line=dict(color="red", width=2, dash="dash")
                                    )

                                    # Update layout for the forest plot
                                    fig_forest.update_layout(
                                        title="Forest Plot of Hazard Ratios",
                                        xaxis=dict(
                                            title="Hazard Ratio",
                                            type="log",  # Log scale for better visualization
                                            range=[np.log10(x_min), np.log10(x_max)],
                                            zeroline=False
                                        ),
                                        yaxis=dict(
                                            title="Variables",
                                            tickvals=list(range(len(summary_df))),
                                            ticktext=summary_df.index,  # Use index names (variables) as y-axis labels
                                            autorange="reversed"  # Reverse Y-axis to match conventional forest plot style
                                        ),
                                        width=800,
                                        height=600,
                                        template="plotly_white"
                                    )

                                    # Display the Plotly figure in Streamlit
                                    st.plotly_chart(fig_forest)


                                except Exception as e:
                                    st.error(f"모델 학습 중 오류가 발생했습니다: {e}")

                                    st.error("자세한 오류 정보: ", traceback.format_exc())  # 스택 트레이스 출력

            except Exception as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
                st.error("자세한 오류 정보: ", traceback.format_exc())  # 스택 트레이스 출력
            except ValueError as e:
                st.error(f"오류가 발생하였으므로 보고가 필요합니다, 문의해주시면 감사하겠습니다.\n: {str(e)}")
                st.error("자세한 오류 정보: ", traceback.format_exc())  # 스택 트레이스 출력
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")


    elif page == "⛔ 오류가 발생했어요":
        # SendGrid API 키 및 이메일 설정
        SENDGRID_API_KEY = "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"
        MY_EMAIL = "hui135@snu.ac.kr"  # 자신의 이메일 주소

        # 이메일 전송 함수
        def send_email_via_sendgrid(subject, content):
            try:
                # 이메일 구성
                email = Mail(
                    from_email="hui135@snu.ac.kr",  # 발신자 이메일
                    to_emails=MY_EMAIL,                  # 수신자 이메일
                    subject=subject,
                    html_content=f"<strong>{content}</strong>"
                )

                # SendGrid 클라이언트를 사용하여 이메일 전송
                sg = SendGridAPIClient(SENDGRID_API_KEY)
                response = sg.send(email)
                return response  # 응답 반환
            except Exception as e:
                st.error(f"Error: {e}")
                return None

        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">⛔ 오류가 발생했어요</h2>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.divider()
        st.write(" ")

        # Get user input
        st.markdown("<h4 style='color:grey;'>어떤 어려움이 있으셨나요?</h4>", unsafe_allow_html=True)
        user_input = st.text_area("여기에 겪고계신 어려움을 작성해주세요. 입력하신 메세지는 김희연 연구원에게 전달됩니다.", key="user_input")

        # 제출 버튼 클릭 시 동작
        if st.button("제출", key="submit_button_1"):
            if user_input.strip() == "":  # 빈 입력 확인
                st.warning("제출 전 내용을 작성해주세요.")
            else:
                # 이메일 전송 시도
                response = send_email_via_sendgrid("User Feedback", user_input)

                if response is None:
                    st.error("전송에 실패하였습니다.")  # 요청 오류 발생 시
                else:
                    # 응답 상태 코드 확인
                    if response.status_code == 202:  # 202는 SendGrid 성공 상태 코드
                        st.success("성공적으로 전송되었습니다.")
                    else:
                        st.error(f"Send failed: {response.text}")
                        st.write(f"Status code: {response.status_code}")

else:
    display_header()
    st.info('GC DataRoom 시스템 이용을 위해선 좌측 사이드바에서 로그인이 필요합니다.', icon="🔔")
