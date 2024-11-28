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
# from causallearn.utils.GraphUtils import graph_to_adjacency_matrix

# wide format
st.set_page_config(layout="wide")

############################
######### Homepage #########
############################

# Image URL
image_url = 'http://www.snuh.org/upload/about/hi/15e707df55274846b596e0d9095d2b0e.png'
title_html = "<h2 class='title'>🏥 헬스케어연구소 🏥 연구자 지원</h2>"
contact_info_html = """
<div style='text-align: right; font-size: 20px; color: grey;'>
오류 문의: 헬스케어연구소 데이터 연구원 김희연 (hui135@snu.ac.kr)
</div>
"""

# Password for accessing the site
PASSWORD = "snuhgchc"  # Change this to your desired password


def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False  # Initialize login state

    if not st.session_state.logged_in:  # Show login form only if not logged in
        st.sidebar.title("로그인")
        password = st.sidebar.text_input("비밀번호를 입력하세요", type="password")
        # Check if the password matches
        if password == PASSWORD:
            st.sidebar.success("비밀번호가 일치합니다")
            st.session_state.logged_in = True  # Set login state to True
            st.session_state.loading_complete = False  # Reset loading state
            return True
        elif password:  # Show an error message only if some input is provided
            st.sidebar.error("비밀번호가 일치하지 않습니다")
        return False
    return True

# Main app logic
if login():  # If logged in, show the rest of the app
    if "loading_complete" not in st.session_state or not st.session_state.loading_complete:
        # Display loading image and spinner for the 3-second delay
        loading_image = st.image(image_url)  # Replace with your loading image path
        with st.spinner("Loading..."):
            time.sleep(3)  # Simulate loading time
        # After loading is complete, remove loading components
        loading_image.empty()
        st.session_state.loading_complete = True  # Mark loading as complete

    # After the loading is complete, show the sidebar menu and hide login form
    st.sidebar.empty()  # Clear the sidebar completely

    # Sidebar with functionality options after login
    st.sidebar.title("기능 선택")
    page = st.sidebar.selectbox(
        "사용하실 기능을 선택해주세요.",
        ["-- 선택 --", "ℹ️ 사용설명서", "📁 피봇 변환", "📈 시각화", "📊 특성표 산출", "🔃 인과관계 추론", "📝 판독문 코딩", "💻 로지스틱 회귀분석", "💻 생존분석", "🎖️ H-PEACE 데이터 파악", "⛔ 오류가 발생했어요"],
        index=0  # Default to "-- 선택 --"
    )

    # Display content based on the page selected
    if page == "-- 선택 --":
        st.markdown(
            """
            <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
                <h2 style="color: #000000;">💡 환영합니다!</h2>
                <p style="font-size:18px; color: #000000;">
                &nbsp;&nbsp;&nbsp;&nbsp;좌측 사이드바에서 원하시는 기능을 선택해주세요.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()
        # Create a checkbox to toggle visibility
        toggle = st.checkbox("Update 사항 자세히보기 - 24.12.11 Updated")
        if toggle:
            st.write("피봇 기능 Age 1 - Age 2- Age 3 등 확인할 것")
            st.write("파일을 읽는 중 오류가 발생했습니다: There are multiple radio elements with the same auto-generated ID")
            st.write("시각화 기능 TypeError: pie() got an unexpected keyword argument 'x'")
            st.write("- (예시) 2024.12.01 📁 피봇 변환 : 오류 수정")
            st.write("- (예시) 2024.12.11 📈 시각화 : 기능 추가")

    elif page == "ℹ️ 사용설명서":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">ℹ️  사용설명서</h2>
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
        selected = st.selectbox("사용설명서를 보실 기능을 선택해주세요.", options=["-- 선택 --", "📁 피봇 변환", "📈 시각화", "📊 특성표 산출", "💻 로지스틱 회귀분석", "💻 생존분석", "🎖️ H-PEACE 데이터 파악", "📝 판독문 코딩"])
        if selected == "-- 선택 --":
            st.write()
        elif selected == "📝 판독문 코딩":
            st.video("https://youtu.be/uE45G40TnTE")

    elif page == "🎖️ H-PEACE 데이터 파악":
        # 네이버 클라우드 API 인증 정보
        access_key = "ncp_iam_BPAMKR52lve6ioI12iS1"  # 발급받은 Access Key ID
        secret_key = "ncp_iam_BPKMKRWniGGEaLImGCq5UB9EkgEQEa7XWV"  # 발급받은 Secret Key


        # 네이버 클라우드 Object Storage의 S3 호환 엔드포인트 설정
        endpoint_url = "https://kr.object.ncloudstorage.com"

        # boto3 클라이언트 생성
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=Config(signature_version='s3v4')  # S3 호환 인증 방식 사용
        )

        # 버킷의 객체 목록 가져오기
        bucket_name = "snuhgc"
        object_name = "hpeace_sample.xlsx"

        try:
            # S3에서 객체(파일)를 가져옵니다.
            response = s3.get_object(Bucket=bucket_name, Key=object_name)

            # 파일 내용을 메모리에 로드
            excel_data = BytesIO(response['Body'].read())

            # Pandas를 사용해 Excel 파일을 DataFrame으로 읽기
            df = pd.read_excel(excel_data, engine='openpyxl')

            # Streamlit에 DataFrame 출력
            st.markdown(
                """
                <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
                    <h2 style="color: #000000;">🎖️ H-PEACE 데이터 파악</h2>
                    <p style="font-size:18px; color: #000000;">
                    &nbsp;&nbsp;&nbsp;&nbsp;H-PEACE 데이터를 살펴보세요.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
                )
            st.divider()
            st.write(" ")

            # Select all df related to the selected patient
            patient_id = st.selectbox('자료를 보실 환자를 선택해주세요.', ["-- 선택 --"] + list(df['GCID'].unique()))
            patient_df = st.session_state.df[st.session_state.df['GCID'] == patient_id]

            cat = ['고혈압_여부',
            '고혈압_투약여부',
            '당뇨_여부',
            '당뇨_투약여부',
            '고지혈증_여부',
            '고지혈증_투약여부',
            '협심증/심근경색증_여부',
            '협심증/심근경색증_투약여부',
            '협심증/심근경색증_중재수술여부_스텐트',
            '협심증/심근경색증_수술여부',
            '뇌졸중(중풍)_여부',
            '뇌졸중(중풍)_투약여부',
            '만성신장염/만성신부전_여부',
            '만성신장염/만성신부전_투약여부',
            '만성신장염/만성신부전_신기능저하여부',
            '만성신장염/만성신부전_투석여부',
            '간경변_여부',
            '간경변_투약여부',
            '만성B형_여부',
            '만성B형_투약여부',
            '만성C형_여부',
            '만성C형_투약여부',
            '폐결핵_여부',
            '폐결핵_투약여부',
            '폐결핵_완치여부_치료종결',
            '폐결핵_반흔여부',
            '천식_여부',
            '천식_투약여부',
            '비염_여부',
            '비염_투약여부',
            '고혈압_통합',
            '당뇨_통합',
            '고지혈증_통합',
            '협심증/심근경색증_통합',
            '뇌졸중(중풍)_통합',
            '만성신장염/만성신부전_통합',
            '간경변_통합',
            '폐결핵_통합',
            '천식_통합',
            '폐암_여부',
            '위암_여부',
            '대장암/직장암_여부',
            '간암_여부',
            '유방암_여부',
            '자궁경부암_여부',
            '갑상선암_여부',
            '전립선암_여부',
            '기타암_여부',
            '고혈압_가족력',
            '당뇨_가족력',
            '만성간염/간경변_가족력',
            '뇌졸중(중풍)_가족력',
            '협심증/심근경색증_가족력',
            '폐암_가족력',
            '위암_가족력',
            '대장암/직장암_가족력',
            '간암_가족력',
            '항혈소판제제_복약여부',
            '항응고제_복약여부',
            '부정맥약_복약여부',
            '인슐린주사/펌프_복약여부',
            '진정제/수면제_복약여부',
            '항우울제/정신과약물_복약여부',
            '갑상선약_복약여부',
            '갑상선기능항진증약_복약여부',
            '골다공증약_복약여부',
            '기타약_복약여부',
            '스테로이드제_복약여부',
            '소염진통제_복약여부',
            '한약_복약여부',
            '칼슘제_복약여부',
            '일반담배_흡연여부',
            '일반담배_과거흡연량',
            '일반담배_현재흡연량',
            '액상형전자담배_흡연여부',
            '궐련형전자담배_흡연여부',
            '고강도_신체활동여부',
            '중강도_신체활동여부',
            '저강도_신체활동여부',
            '고강도_운동여부',
            '중강도_운동여부']

            num = [
                '일반담배_과거흡연기간',
                '일반담배_현재흡연기간',
                '액상형전자담배_현재흡연빈도',
                '궐련형전자담배_현재흡연량',
                '음주빈도',
                '음주량',
                '고강도_신체활동빈도',
                '고강도_신체활동시간',
                '중강도_신체활동빈도',
                '중강도_신체활동시간',
                '저강도_신체활동빈도',
                '저강도_신체활동시간',
                '고강도_운동빈도',
                '고강도_운동시간',
                '중강도_운동빈도',
                '중강도_운동시간',
                '최종학력',
                '결혼상태',
                '가계수입']

            # Function to get a combined view of past and current df
            def combined_status(patient_df, cat):
                cat_columns = [col for col in patient_df.columns if cat in col]
                combined_results = {}

                for disease in cat_columns:
                    if disease in patient_df.columns:
                        unique_vals = patient_df[disease].unique()
                        # Initialize an empty string to store check marks, crosses, or question marks in a stacked format
                        combined_results[disease] = ""

                        # Iterate through unique values
                        for val in unique_vals:
                            if pd.isna(val):  # Check if the value is NaN
                                combined_results[disease] += "❔ "
                            elif val == 1:
                                combined_results[disease] += "✔️ "
                            else:
                                combined_results[disease] += "❌ "

                return combined_results

            # Display combined information
            def display_combined_info(combined_results):
                cols = st.columns(3)  # Create 3 columns to distribute the information
                for i, (disease, status) in enumerate(combined_results.items()):
                    cols[i % 3].markdown(f"**{disease}**: {status}")

            # Function to plot lab changes over multiple visits
            def plot_changes(patient_df, num):
                # Extract num columns (assuming num columns follow a specific naming pattern)
                num_columns = [col for col in patient_df.columns if num in col]
                num_df = []

                # Collect num df across visits
                for col in num_columns:
                    # Iterate through all rows (assuming each row represents a visit)
                    for i in range(len(patient_df)):
                        if pd.notna(patient_df[col].iloc[i]):
                            num_df.append((i + 1, patient_df[col].iloc[i]))  # (visit number, num value)

                # Plotting num changes
                if num_df:
                    visits, num_values = zip(*num_df)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=visits,
                        y=num_values,
                        mode='lines+markers',
                        name=num,
                        marker=dict(symbol='circle', size=10),
                    ))

                    fig.update_xaxes(title_text="Visits", dtick=1)
                    fig.update_yaxes(title_text=f"{num} Levels")

                    fig.update_layout(height=600, width=900, title_text=f"{num}", showlegend=False)

                    return fig
                else:
                    return None

            no_data_messages = []

            for col in cat:
                comorbidity_results = combined_status(patient_df, col)
                display_combined_info(comorbidity_results)

            for col in num:
                fig = plot_changes(patient_df, col)
                if fig:
                    st.plotly_chart(fig)

            # # Loop through columns and display the results for categorical and numerical data separately
            # for col in patient_df.columns if col == :
            #     if patient_df[col].nunique() == 2:  # Checking for categorical columns with two unique values
            #         if patient_df[col].isnull().all():  # Check if all values in the column are NaN
            #             no_data_messages.append(f"No meaningful data for {col} in this patient (all values are NaN).")
            #         else:
            #             comorbidity_results = combined_status(patient_df, col)
            #             display_combined_info(comorbidity_results)

            # for col in patient_df.columns:
            #     if patient_df[col].nunique() >= 3:  # Checking for numerical columns with three or more unique values
            #         fig = plot_changes(patient_df, col)
            #         if fig:
            #             st.plotly_chart(fig)
            #         else:
            #             no_data_messages.append(f"No {col} data available for this patient.")

            # Display all "No data available" messages if any
            if no_data_messages:
                for message in no_data_messages:
                    st.write(message)

        except Exception as e:
            st.write("파일을 불러오는 중 오류가 발생했습니다:", e)

    elif page == "📁 피봇 변환":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📁 피봇 변환</h2>
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
        uploaded_file = st.file_uploader("환자의 여러 내원 데이터가 행으로 축적되어있는 데이터 파일을 올려주세요.", type=["csv", "xlsx"])


        if uploaded_file is not None:
            # 파일이 새로 업로드되었을 때 세션 상태 초기화
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if st.session_state.uploaded_file != uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.phrases_by_code = {}
                    st.session_state.text_input = ""
                    st.session_state.code_input = ""
                    st.session_state.coded_df = None  # Initialize session state for coded DataFrame

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="📁 피봇 변환")

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
                id_column = st.selectbox("환자를 구분할 ID 혹은 RID 열을 선택해주세요.", ["-- 선택 --"] + list(df.columns))
                if id_column == "-- 선택 --":
                    st.write(" ")
                    st.stop()

                date_column = st.selectbox("방문을 구분할 Date 열을 선택해주세요.", ["-- 선택 --"] + list(df.columns))
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
                st.header("📁 피봇 변환 결과", divider='rainbow')

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

                # 세션 상태에 저장
                st.session_state.df_pivot = df_pivot
                st.divider()
                st.markdown("<h4 style='color:grey;'>피봇 데이터</h4>", unsafe_allow_html=True)
                st.dataframe(df_pivot)

                # Initialize session state for df_pivot and button states if they are not already present
                if "df_pivot" not in st.session_state:
                    st.session_state.df_pivot = df_pivot  # Load df_pivot data if not already loaded
                if "filter_button_pressed" not in st.session_state:
                    st.session_state.filter_button_pressed = False
                if "download_button_pressed" not in st.session_state:
                    st.session_state.download_button_pressed = False
                if "download_filtered_button_pressed" not in st.session_state:
                    st.session_state.download_filtered_button_pressed = False

                # Display additional functionalities only if df_pivot is present in the session state
                if st.session_state.df_pivot is not None:
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>추가 작업 수행</h4>", unsafe_allow_html=True)

                    # Filter button in a row
                    if st.button("재진 필터링"):
                        # Update session state to indicate that the filter button was pressed
                        st.session_state.filter_button_pressed = True

                    # Filter functionality with dynamic column dropping if filter_button is pressed
                    if st.session_state.filter_button_pressed:
                        num = st.selectbox("최대 내원 횟수를 n회로 지정합니다.", ["-- 선택 --"] + list(range(1, max_len)))

                        if num != "-- 선택 --":
                            # Determine columns to keep based on the selected max visit count
                            columns_to_keep = [col for col in st.session_state.df_pivot.columns
                                            if not any(col.endswith(f"_{i}") for i in range(num + 1, max_len + 1))]

                            # Filter the DataFrame based on selected columns
                            df_pivot_filtered = st.session_state.df_pivot[columns_to_keep]
                            st.session_state.df_pivot_filtered = df_pivot_filtered

                            # Display the filtered DataFrame
                            st.dataframe(df_pivot_filtered, use_container_width=True)

                            export_format_filtered = st.radio("다운로드하실 자료 파일 형식을 선택하세요.", options=["CSV", "Excel"], key="export_format_filtered")

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

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 다운로드</h4>", unsafe_allow_html=True)

                    # Display the file format selection radio button for original data download
                    export_format_original = st.radio("다운로드하실 자료 파일 형식을 선택하세요.", options=["CSV", "Excel"], key="export_format_original")

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

            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")

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
        uploaded_file = st.file_uploader("판독문 텍스트 열을 포함한 파일을 업로드하세요.", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # 파일이 새로 업로드되었을 때 세션 상태 초기화
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if st.session_state.uploaded_file != uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.phrases_by_code = {}
                    st.session_state.text_input = ""
                    st.session_state.code_input = ""
                    st.session_state.coded_df = None  # Initialize session state for coded DataFrame

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                # 데이터 미리보기 표시
                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="📝 판독문 코딩")

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
                    column_selected = st.selectbox("코딩할 판독문 텍스트 열을 선택하세요.", options=df.columns)

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
                        <p>하단에 코드와 함께 텍스트를 입력 시, 해당 텍스트가 포함된 판독문 행은 함께 입력된 코드로 코딩이 이뤄집니다.</p>
                        <p> </p>
                        <p>🔔 주의!) 먼저 입력한 코드 내용보다 뒤에 입력한 코드 내용에 높은 우선순위가 부여됩니다.</p>
                        <p>    - Case 1) 코드 1과 "disease1" 입력 후, 코드 2와 다시 "disease1" 입력: "disease1"이 포함된 행은 2로 코딩됩니다.</p>
                        <p>    - Case 2) 코드 1과 "disease1" 입력 후, 코드 2와 "disease2" 입력: "disease1, disease2" 모두 포함된 행은 2로 코딩됩니다.</p>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write(" ")

                current_code = st.text_input("코드를 입력하고 엔터를 누르세요. (ex - 0, 1, 2)", key="code_input")

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
                    st.text_input("텍스트를 입력하고 엔터를 누르세요.", key="text_input", on_change=add_text)

                # 4. 입력된 코딩 및 텍스트 목록 표시 및 삭제 기능 추가
                if st.session_state.phrases_by_code:
                    st.write(" ")
                    st.write(" ")
                    st.markdown("<h5>현재 입력된 코드 및 텍스트 목록 :</h5>", unsafe_allow_html=True)
                    for code, phrases in st.session_state.phrases_by_code.items():
                        st.markdown(f"<span style='color:red;'>코드 {code}에 대한 텍스트:</span>", unsafe_allow_html=True)
                        # Create a dynamic list where phrases can be deleted
                        for phrase in phrases:
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"<span style='color:red;'>- {phrase}</span>", unsafe_allow_html=True)
                            with col2:
                                if st.button(f"삭제", key=f"delete_{code}_{phrase}"):
                                    st.session_state.phrases_by_code[code].remove(phrase)  # Remove the phrase from the list
                                    # Force rerun by altering a session state value
                                    st.session_state["rerun_trigger"] = not st.session_state.get("rerun_trigger", False)

                # 5. 미처리 항목을 자동으로 0으로 처리 또는 다른 방식 처리
                st.divider()
                st.markdown("<h4 style='color:grey;'>코딩되지 않은 그 외 판독문 처리 방법</h4>", unsafe_allow_html=True)

                # Use radio buttons to select between filling with 0 or missing
                fill_option = st.radio("처리 방법을 선택하세요.", ("전부 0으로", "전부 99로", "전부 공백으로"))

                # 3. 완료 버튼 - 텍스트 입력 후 활성화
                st.divider()
                st.markdown("<h4 style='color:grey;'>코딩 작업 종료하기</h4>", unsafe_allow_html=True)
                if current_code and st.session_state.phrases_by_code[current_code]:
                    if st.button("코딩 종료"):
                        # Create a temporary lowercase column for matching
                        df = st.session_state.df.copy()  # Use session_state to preserve df between runs
                        df['lower_temp'] = df[column_selected].str.lower()

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
                        st.write("코딩이 완료되었습니다.")
                        st.dataframe(st.session_state.coded_df, use_container_width=True)

                    # 6. 데이터 다운로드 버튼 (Excel 또는 CSV)
                    if st.session_state.coded_df is not None:
                        st.divider()
                        st.markdown("<h4 style='color:grey;'>데이터 다운로드</h4>", unsafe_allow_html=True)
                        export_format = st.radio("파일 형식을 선택하세요", options=["CSV", "Excel"])
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

            except ValueError as e:
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "📈 시각화":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📈 시각화</h2>
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
        uploaded_file = st.file_uploader("시각화에 이용하실 데이터 파일을 업로드해주세요.")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="📈 시각화")

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
                    st.header("📈 Univariable 데이터 시각화", divider='rainbow')
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
                    plot_type = st.radio("그래프 선택", ('범주형 변수 : Barplot', '범주형 변수 : Pie chart', '연속형 변수 : Histogram', '연속형 변수 : Boxplot'))
                    st.text("")

                    # Creating visualizations using Plotlyif plot_type == '범주형 변수 : Barplot':
                    # Convert to categorical data if necessary
                    # Create visualizations based on user's selection
                    if plot_type:  # 사용자가 시각화 유형을 선택하면 실행
                        if plot_type == '범주형 변수 : Barplot':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")

                            selected_column = st.selectbox("열 선택", columns)
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

                            selected_column = st.selectbox("열 선택", columns)
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

                            selected_column = st.selectbox("열 선택", columns)
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

                            selected_column = st.selectbox("열 선택", columns)
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
                    st.header("📈 Multivariable 데이터 시각화", divider='rainbow')
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
                    plot_type = st.radio("그래프 선택", ('범주형 변수 : Barplot', '범주형 변수 : Pie chart', '연속형 변수 : Histogram', '연속형 변수 : Boxplot', '연속형 변수: Correlation Heatmap'))
                    st.text("")

                    # Creating visualizations using Plotlyif plot_type == '범주형 변수 : Barplot':
                    # Convert to categorical data if necessary
                    # Create visualizations based on user's selection
                    if plot_type:  # 사용자가 시각화 유형을 선택하면 실행
                        if plot_type == '범주형 변수 : Barplot':
                            # Convert to categorical data if necessary
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("열 선택", columns, key='categorical_variable')
                            selected_column_2 = st.selectbox("그룹열 선택", columns, key='group_variable')

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
                            selected_column_1 = st.selectbox("열 선택", columns, key='categorical_variable')
                            selected_column_2 = st.selectbox("그룹열 선택", columns, key='group_variable')

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
                            selected_column_1 = st.selectbox("열 선택", columns, key='continuous_variable')
                            selected_column_2 = st.selectbox("그룹열 선택", columns, key='group_variable')

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
                                    st.warning("Histogram은 연속형 변수에 적합합니다. 다른 열을 선택해주세요.")

                        elif plot_type == '연속형 변수 : Boxplot':
                            # Check if the selected column is continuous
                            columns = df.columns.tolist()
                            columns.insert(0, "-- 선택 --")
                            selected_column_1 = st.selectbox("열 선택", columns, key='continuous_variable')
                            selected_column_2 = st.selectbox("그룹열 선택", columns, key='group_variable')

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
                                    st.warning("Boxplot은 연속형 변수에 적합합니다. 다른 열을 선택해주세요.")

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
                            st.write("먼저 시각화 유형을 선택해주세요.")

            except ValueError as e:
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "📊 특성표 산출":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">📊 특성표 산출</h2>
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
        uploaded_file = st.file_uploader("특성표 산출에 이용하실 데이터 파일을 업로드해주세요.")
        st.warning("업로드 시, 날짜형 타입의 열은 자동으로 인식하여 제외됩니다.", icon="🚨")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="📊 특성표 산출")

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
                    st.markdown("<h4 style='color:grey;'>특성표 생성</h4>", unsafe_allow_html=True)
                    st.header("📊 특성표 생성", divider="rainbow")

                    # Let the user select whether the dependent variable has 2 or 3 categories
                    category_choice = st.radio(
                        "종속변수가 몇 개의 범주를 가지는지 선택해주세요.",
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
                            "통계적 유의성을 볼 종속변수를 선택해주세요.",
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
                            st.markdown("<h4 style='color:grey;'>데이터 다운로드</h4>", unsafe_allow_html=True)
                            export_format = st.radio("파일 형식을 선택하세요", options=["CSV", "Excel"])
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
                            "통계적 유의성을 볼 종속변수를 선택해주세요.",
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
                            export_format = st.radio("파일 형식을 선택하세요", options=["CSV", "Excel"])
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
            except ValueError as e:
                st.error(e)
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    elif page == "🔃 인과관계 추론":
        st.markdown(
        """
        <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
            <h2 style="color: #000000;">🔃 인과관계 추론</h2>
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
        uploaded_file = st.file_uploader("판독문 텍스트 열을 포함한 파일을 업로드하세요.", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # 파일이 새로 업로드되었을 때 세션 상태 초기화
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if st.session_state.uploaded_file != uploaded_file:
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.phrases_by_code = {}
                    st.session_state.text_input = ""
                    st.session_state.code_input = ""
                    st.session_state.coded_df = None  # Initialize session state for coded DataFrame

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                # 데이터 미리보기 표시
                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="🔃 인과관계 추론")

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
                    st.markdown("<h4 style='color:grey;'>관계를 볼 변수 선택</h4>", unsafe_allow_html=True)
                    column_selected = st.selectbox("인과관계를 볼 변수 열을 선택하세요.", options=df.columns)

                    # 선택된 열을 기반으로 작업
                    st.session_state.df = df  # Ensure df is stored initially

                    def get_categorical_columns(df):
                        categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
                        low_cardinality_numerical = [col for col in df.select_dtypes(exclude=['object', 'category']).columns if df[col].nunique() < 5]
                        return categorical_columns + low_cardinality_numerical

                    # Separate selections for continuous and categorical variables
                    continuous_columns = st.multiselect(
                        "- 연속형 설명변수(X)를 선택해주세요.",
                        df.select_dtypes(include=['float64', 'int64']).columns,
                        key="continuous_columns_selection"
                    )

                    categorical_columns = st.multiselect(
                        "- 범주형 설명변수(X)를 선택해주세요.",
                        get_categorical_columns(df),
                        key="categorical_columns_selection"
                    )

                    st.session_state.X_columns = continuous_columns + categorical_columns
                    if 'proceed_to_preprocessing' not in st.session_state:
                        st.session_state.proceed_to_preprocessing = False

                    # Add a button to confirm the selections
                    if st.button('선택 완료', key='complete_button'):
                        if (continuous_columns or categorical_columns):  # Ensure that y and at least one X is selected
                            st.session_state.continuous_columns = continuous_columns
                            st.session_state.categorical_columns = categorical_columns
                            st.session_state.proceed_to_preprocessing = True
                        else:
                            st.warning("변수를 한 개 이상 선택해주세요.", icon="⚠️")

                    # Check if preprocessing should proceed
                    if st.session_state.proceed_to_preprocessing:
                        st.divider()
                        st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                        # Check missing values for continuous and categorical variables separately
                        for X_column in st.session_state.continuous_columns:
                            X_missing_count = df[X_column].isna().sum()
                            st.markdown(
                                f"<p style='font-size:16px; color:firebrick;'><strong>선택된 연속형 설명변수 '{X_column}'에 결측이 {X_missing_count}개 존재합니다.</strong></p>",
                                unsafe_allow_html=True
                            )

                        for X_column in st.session_state.categorical_columns:
                            X_missing_count = df[X_column].isna().sum()
                            st.markdown(
                                f"<p style='font-size:16px; color:firebrick;'><strong>선택된 범주형 설명변수 '{X_column}'에 결측이 {X_missing_count}개 존재합니다.</strong></p>",
                                unsafe_allow_html=True
                            )

                        # Function to handle missing values
                        def handle_missing_values(df, columns, strategies):
                            for column, strategy in strategies.items():
                                if strategy == '결측이 존재하는 행을 제거':
                                    df = df.dropna(subset=[column])
                                elif strategy in ['해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체']:
                                    impute_strategy = {
                                        '해당 열의 평균값으로 대체': 'mean',
                                        '해당 열의 중앙값으로 대체': 'median',
                                        '해당 열의 최빈값으로 대체': 'most_frequent'
                                    }[strategy]
                                    imputer = SimpleImputer(strategy=impute_strategy)
                                    df[[column]] = imputer.fit_transform(df[[column]])
                            return df

                        # Synchronize indexes between continuous and categorical data
                        def synchronize_indexes(X_continuous, X_categorical):
                            shared_indexes = X_continuous.index.intersection(X_categorical.index)
                            return X_continuous.loc[shared_indexes], X_categorical.loc[shared_indexes]

                        # Separate continuous and categorical columns
                        X_continuous = df[st.session_state.continuous_columns]
                        X_categorical = df[st.session_state.categorical_columns]

                        # Check for missing values
                        continuous_missing = X_continuous.isnull().any().any()
                        categorical_missing = X_categorical.isnull().any().any()

                        if not continuous_missing and not categorical_missing:
                            st.markdown(
                                f"<p style='font-size:16px;'><strong>결측 처리 작업 없이 분석이 가능합니다.</strong></p>",
                                unsafe_allow_html=True
                            )

                        else:
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>변수 전처리</h4>", unsafe_allow_html=True)

                            # Missing value strategies for continuous columns
                            continuous_missing_value_strategies = {}
                            for column in st.session_state.continuous_columns:
                                if df[column].isnull().any():
                                    n = df[column].isna().sum()
                                    strategy = st.selectbox(
                                        f"- ⚠️ 선택하신 연속형 설명변수 '{column}'에 {n}개의 결측이 있습니다. 처리 방법을 선택하세요:",
                                        ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                        key=f"{column}_strategy"
                                    )
                                    if strategy != '-- 선택 --':
                                        continuous_missing_value_strategies[column] = strategy

                            # Missing value strategies for categorical columns
                            categorical_missing_value_strategies = {}
                            for column in st.session_state.categorical_columns:
                                if df[column].isnull().any():
                                    n = df[column].isna().sum()
                                    strategy = st.selectbox(
                                        f"- ⚠️ 선택하신 범주형 설명변수 '{column}'에 {n}개의 결측이 있습니다. 처리 방법을 선택하세요:",
                                        ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                        key=f"{column}_strategy"
                                    )
                                    if strategy != '-- 선택 --':
                                        categorical_missing_value_strategies[column] = strategy

                            # Apply missing value strategies
                            if continuous_missing_value_strategies:
                                X_continuous = handle_missing_values(X_continuous, st.session_state.continuous_columns, continuous_missing_value_strategies)

                            if categorical_missing_value_strategies:
                                X_categorical = handle_missing_values(X_categorical, st.session_state.categorical_columns, categorical_missing_value_strategies)

                            # Synchronize indexes between X_continuous and X_categorical
                            X_continuous, X_categorical = synchronize_indexes(X_continuous, X_categorical)

                        st.divider()
                        st.header("🔃 인과관계 추론", divider="rainbow")

                        # Example: Combine continuous and categorical data
                        X = pd.concat([X_continuous, X_categorical], axis=1)

                        # Step 1: Run the PC algorithm to learn causal structure
                        cg = pc(X.to_numpy(), alpha=0.05)

                        # Step 2: Extract causal edges
                        def extract_edges(causal_graph, column_names):
                            edges = []
                            for i in range(len(causal_graph)):
                                for j in range(len(causal_graph)):
                                    if causal_graph[i, j] == 1:  # Direction i → j
                                        edges.append((column_names[i], column_names[j]))
                                    elif causal_graph[i, j] == -1:  # Direction j → i
                                        edges.append((column_names[j], column_names[i]))
                            return edges

                        edges = extract_edges(cg.G.graph, X.columns)

                        # Step 3: Create a NetworkX graph
                        causal_graph = nx.DiGraph()
                        causal_graph.add_edges_from(edges)

                        # Step 4: Visualize the graph with proper padding
                        def visualize_graph(graph, seed=None, padding_ratio=0.05):
                            # Generate positions for nodes
                            pos = nx.spring_layout(graph, seed=seed)

                            # Extract node positions and edges
                            edge_x = []
                            edge_y = []
                            annotations = []

                            for edge in graph.edges():
                                x0, y0 = pos[edge[0]]  # Start node position
                                x1, y1 = pos[edge[1]]  # End node position

                                # Calculate vector components and length
                                dx, dy = x1 - x0, y1 - y0
                                dist = (dx**2 + dy**2)**0.5

                                # Apply padding to both start and end points
                                x0_padded = x0 + dx * padding_ratio / dist
                                y0_padded = y0 + dy * padding_ratio / dist
                                x1_padded = x1 - dx * padding_ratio / dist
                                y1_padded = y1 - dy * padding_ratio / dist

                                # Add edges for visual reference
                                edge_x.append(x0_padded)
                                edge_x.append(x1_padded)
                                edge_x.append(None)
                                edge_y.append(y0_padded)
                                edge_y.append(y1_padded)
                                edge_y.append(None)

                                # Add arrow annotations for direction
                                annotations.append(
                                    dict(
                                        ax=x0_padded, ay=y0_padded,  # Adjusted start point
                                        x=x1_padded, y=y1_padded,  # Adjusted end point
                                        xref="x", yref="y",
                                        axref="x", ayref="y",
                                        showarrow=True,
                                        arrowhead=3,  # Arrow style
                                        arrowsize=1.5,  # Arrow size
                                        arrowwidth=1.5,  # Arrow line width
                                        arrowcolor="gray"
                                    )
                                )

                            # Create edge traces
                            edge_trace = go.Scatter(
                                x=edge_x,
                                y=edge_y,
                                line=dict(width=1.5, color='gray'),
                                hoverinfo='none',
                                mode='lines'
                            )

                            # Create node traces
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
                                textfont=dict(family='Times New Roman', size=12, color='darkblue'),
                                marker=dict(
                                    size=30,
                                    color='lightblue',
                                    line=dict(width=2, color='darkblue')
                                ),
                                hoverinfo='text'
                            )

                            # Combine traces
                            fig = go.Figure(data=[edge_trace, node_trace])

                            # Add arrow annotations for edges
                            fig.update_layout(
                                showlegend=False,
                                title_text="Causal Graph with Padded Nodes and Arrows",
                                title_font=dict(family="Times New Roman", size=20, color="darkblue"),
                                margin=dict(l=40, r=40, t=40, b=40),
                                xaxis=dict(showgrid=False, zeroline=False),
                                yaxis=dict(showgrid=False, zeroline=False),
                                annotations=annotations  # Add arrows
                            )

                            # Display the graph in Streamlit
                            st.plotly_chart(fig)

                        # Visualize the graph
                        if "random_seed" not in st.session_state:
                            st.session_state.random_seed = 99  # Default seed

                        visualize_graph(causal_graph, seed=st.session_state.random_seed)

                        # Add a regenerate button to update the layout
                        if st.button("Regenerate Layout"):
                            st.session_state.random_seed = None  # Clear the seed for a new random layout
                            visualize_graph(causal_graph, seed=st.session_state.random_seed)

                        # hc = HillClimbSearch(X)
                        # model = hc.estimate(scoring_method=BicScore(X))  # Bayesian Information Criterion (BIC)

                        # # Display learned edges
                        # st.write("Learned Causal Edges:", model.edges())

                        # # Step 5: Create the causal graph
                        # causal_graph = nx.DiGraph(model.edges)

                        # # Step 6: Visualize the graph using matplotlib
                        # plt.figure(figsize=(10, 8))
                        # plt.rc('font', family='Times New Roman')  # Set font globally
                        # pos = nx.spring_layout(causal_graph, seed=42)  # Generate positions for nodes
                        # nx.draw(
                        #     causal_graph, pos, with_labels=True,
                        #     node_size=3000,  # Node size
                        #     node_color="lightblue",  # Node color
                        #     font_size=12,  # Node label font size
                        #     font_color="darkblue",  # Node label color
                        #     edge_color="gray",  # Edge color
                        #     arrowsize=20,  # Arrow size
                        #     width=2  # Edge thickness
                        # )

                        # # Add edge labels
                        # edge_labels = {(edge[0], edge[1]): f"{edge[0]}→{edge[1]}" for edge in causal_graph.edges}
                        # nx.draw_networkx_edge_labels(causal_graph, pos, edge_labels=edge_labels, font_size=10)

                        # # Add a title
                        # plt.title("Causal Graph Discovered from X", fontsize=18, color="darkblue", pad=20)

                        # # Step 7: Display the graph in Streamlit
                        # st.pyplot(plt)


            except ValueError as e:
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
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
        uploaded_file = st.file_uploader("로지스틱 회귀분석에 이용하실 데이터 파일을 업로드해주세요.")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        xls = pd.ExcelFile(uploaded_file)
                        sheet_names = xls.sheet_names

                        # 시트 선택 옵션에 "-- 선택 --" 추가
                        if len(sheet_names) > 1:
                            sheet = st.selectbox('업로드하신 파일에 여러 시트가 존재합니다. 이용하실 시트를 선택해주세요.', ["-- 선택 --"] + sheet_names)

                            # "-- 선택 --"인 경우 동작 중단
                            if sheet == "-- 선택 --":
                                st.stop()  # 이후 코드를 실행하지 않도록 중단
                            else:
                                df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        elif len(sheet_names) == 1:
                            # 시트가 1개만 있는 경우
                            df = pd.read_excel(uploaded_file, sheet_name=sheet_names[0])
                        else:
                            st.error("엑셀 파일에 시트가 없습니다.")
                            st.stop()

                if 'df' in locals():

                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="💻 로지스틱 회귀분석")

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
                        "종속변수(y)를 선택해주세요.",
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
                            "- 연속형 설명변수(X)를 선택해주세요.",
                            df.select_dtypes(include=['float64', 'int64']).columns,
                            key="continuous_columns_selection"
                        )

                        categorical_columns = st.multiselect(
                            "- 범주형 설명변수(X)를 선택해주세요.",
                            get_categorical_columns(df),
                            key="categorical_columns_selection"
                        )

                        st.session_state.X_columns = continuous_columns + categorical_columns

                        # Add a button to confirm the selections
                        if st.button('선택 완료', key='complete_button'):
                            if y_column and (continuous_columns or categorical_columns):  # Ensure that y and at least one X is selected
                                st.session_state.y_column = y_column
                                st.session_state.continuous_columns = continuous_columns
                                st.session_state.categorical_columns = categorical_columns
                                st.session_state.proceed_to_preprocessing = True
                            else:
                                st.warning("종속변수와 설명변수를 한 개 이상 선택해주세요.", icon="⚠️")

                        # Check if preprocessing should proceed
                        if st.session_state.proceed_to_preprocessing:
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)
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
                                <p>- 선택하신 종속변수(y)에 결측값이 존재한다면, 해당 행은 분석에서 제외됩니다.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                            )
                            st.write(" ")

                            # Display missing value check for the dependent variable
                            y_missing_count = df[st.session_state.y_column].isna().sum()
                            st.markdown(
                                f"<p style='font-size:16px; color:red;'><strong>선택된 종속변수 '{st.session_state.y_column}'에 결측이 {y_missing_count}개 존재합니다.</strong></p>",
                                unsafe_allow_html=True
                            )

                            # Check missing values for continuous and categorical variables separately
                            for X_column in st.session_state.continuous_columns:
                                X_missing_count = df[X_column].isna().sum()
                                st.markdown(
                                    f"<p style='font-size:16px; color:firebrick;'><strong>선택된 연속형 설명변수 '{X_column}'에 결측이 {X_missing_count}개 존재합니다.</strong></p>",
                                    unsafe_allow_html=True
                                )

                            for X_column in st.session_state.categorical_columns:
                                X_missing_count = df[X_column].isna().sum()
                                st.markdown(
                                    f"<p style='font-size:16px; color:firebrick;'><strong>선택된 범주형 설명변수 '{X_column}'에 결측이 {X_missing_count}개 존재합니다.</strong></p>",
                                    unsafe_allow_html=True
                                )

                            # Drop rows with missing values in the dependent variable (y)
                            df = df.dropna(subset=[st.session_state.y_column])

                            # Separate continuous and categorical columns
                            X_continuous = df[st.session_state.continuous_columns]
                            X_categorical = df[st.session_state.categorical_columns]
                            y = df[st.session_state.y_column]

                            if not X_continuous.isnull().any().any() and not X_categorical.isnull().any().any():
                                st.markdown(
                                    f"<p style='font-size:16px;'><strong>결측 처리 작업 없이 분석이 가능합니다.</strong></p>",
                                    unsafe_allow_html=True
                                )

                            else:

                                st.divider()
                                st.markdown("<h4 style='color:grey;'>변수 전처리</h4>", unsafe_allow_html=True)

                                # Handling missing value strategies for continuous columns
                                continuous_missing_value_strategies = {}
                                for column in st.session_state.continuous_columns:
                                    if df[column].isnull().any():
                                        n = df[column].isna().sum()
                                        strategy = st.selectbox(
                                            f"- ⚠️ 선택하신 연속형 설명변수 '{column}'에 {n}개의 결측이 있습니다. 처리 방법을 선택하세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                            key=f"{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            continuous_missing_value_strategies[column] = strategy

                                # Handling missing value strategies for categorical columns
                                categorical_missing_value_strategies = {}
                                for column in st.session_state.categorical_columns:
                                    if df[column].isnull().any():
                                        n = df[column].isna().sum()
                                        strategy = st.selectbox(
                                            f"- ⚠️ 선택하신 범주형 설명변수 '{column}'에 {n}개의 결측이 있습니다. 처리 방법을 선택하세요:",
                                            ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                            key=f"{column}_strategy"
                                        )
                                        if strategy != '-- 선택 --':
                                            categorical_missing_value_strategies[column] = strategy

                                # Apply missing value strategies for continuous columns
                                for column, strategy in continuous_missing_value_strategies.items():
                                    if strategy == '결측이 존재하는 행을 제거':
                                        X_continuous = X_continuous.dropna(subset=[column])
                                    elif strategy in ['해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체']:
                                        impute_strategy = {
                                            '해당 열의 평균값으로 대체': 'mean',
                                            '해당 열의 중앙값으로 대체': 'median',
                                            '해당 열의 최빈값으로 대체': 'most_frequent'
                                        }[strategy]
                                        imputer = SimpleImputer(strategy=impute_strategy)
                                        X_continuous[[column]] = imputer.fit_transform(X_continuous[[column]])

                                # Ensure continuous columns are numeric
                                for column in X_continuous.columns:
                                    X_continuous[column] = pd.to_numeric(X_continuous[column], errors='coerce')
                                    X_continuous[column] = X_continuous[column].astype(float)

                                # Apply missing value strategies for categorical columns
                                for column, strategy in categorical_missing_value_strategies.items():
                                    if strategy == '결측이 존재하는 행을 제거':
                                        X_categorical = X_categorical.dropna(subset=[column])
                                    elif strategy == '해당 열의 최빈값으로 대체':
                                        imputer = SimpleImputer(strategy='most_frequent')
                                        X_categorical[[column]] = imputer.fit_transform(X_categorical[[column]])

                                # Synchronize index changes between X_continuous, X_categorical, and y
                                shared_indexes = X_continuous.index.intersection(X_categorical.index)
                                X_continuous = X_continuous.loc[shared_indexes]
                                X_categorical = X_categorical.loc[shared_indexes]
                                y = y.loc[shared_indexes]

                            # Check for missing or infinite values in combined data
                            if st.button('모델 학습 시작', key='train_model_button'):
                                # Handle categorical variables with pd.get_dummies
                                st.divider()
                                st.header('💻 로지스틱 회귀분석 결과', divider='rainbow')
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

                                if X.isnull().values.any():
                                    st.error("전처리 후에도 설명변수에 결측치가 남아 있습니다. 결측치 처리를 확인해주세요.")
                                elif np.isinf(X).values.any():
                                    st.error("전처리 후 설명변수에 무한 값이 존재합니다. 데이터 정규화를 확인해주세요.")
                                else:
                                    try:
                                        # Split the data
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                                        # Add constant (intercept) to the features
                                        X_train_const = sm.add_constant(X_train)
                                        X_test_const = sm.add_constant(X_test)

                                        # Final checks for NaN or infinite values before fitting the model
                                        if X_train_const.isnull().values.any() or np.isinf(X_train_const).values.any():
                                            st.error("모델 학습 데이터에 NaN 또는 무한 값이 포함되어 있습니다. 전처리를 확인해주세요.")
                                        elif X_test_const.isnull().values.any() or np.isinf(X_test_const).values.any():
                                            st.error("모델 테스트 데이터에 NaN 또는 무한 값이 포함되어 있습니다. 전처리를 확인해주세요.")
                                        else:
                                            # Logistic regression model using statsmodels
                                            model = sm.Logit(y_train, X_train_const)
                                            result = model.fit()

                                            # Predictions
                                            y_pred_prob = result.predict(X_test_const)
                                            y_pred_class = (y_pred_prob >= 0.5).astype(int)

                                        # Display model summary
                                        # st.markdown("<h3 style='font-size:14px;'>Model Results:</h3>", unsafe_allow_html=True)
                                        # Assuming result is a statsmodels results object
                                        summary_html = result.summary().as_html()

                                        # Display the summary as HTML
                                        # summary_df = result.summary().tables[1]
                                        # st.dataframe(summary_df)

                                        st.markdown(summary_html, unsafe_allow_html=True)
                                        st.write(" ")
                                        st.write("---")

                                        st.markdown("<h4 style='font-size:14px;'>Model OR & P-value:</h4>", unsafe_allow_html=True)
                                        summary_table = result.summary2().tables[1]  # Get the detailed table
                                        summary_df = summary_table[['Coef.', 'P>|z|', '[0.025', '0.975]']]  # Extract Coefficient, p-value, and Confidence Intervals

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

                                        # Classification Report
                                        st.markdown("<h5 style='font-size:14px;'>Classification Report:</h5>", unsafe_allow_html=True)
                                        report = classification_report(y_test, y_pred_class, output_dict=True)
                                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

                                    except Exception as e:
                                        st.error(f"모델 학습 중 오류가 발생했습니다: {str(e)}")

            except ValueError as e:
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
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
        uploaded_file = st.file_uploader("생존분석에 이용하실 데이터 파일을 업로드해주세요.")

        if uploaded_file is not None:
            try:
                # 파일 크기가 큰 경우를 대비해 오류 처리
                if uploaded_file.size > 200 * 1024 * 1024:  # 200MB 제한
                    raise ValueError("파일 크기가 너무 큽니다. 행의 개수 혹은 열의 개수를 줄인 후 다시 업로드해주세요.")

                if uploaded_file:  # 파일이 업로드된 경우에만 실행
                    # 파일 타입에 따라 데이터 읽기
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)

                if 'df' in locals():
                    st.divider()
                    st.markdown("<h4 style='color:grey;'>데이터 미리보기</h4>", unsafe_allow_html=True)

                    # Add a radio button for the user to select the option
                    selected_option = st.radio("옵션을 선택하세요:", ["데이터", "결측수", "요약통계"], key="💻 생존분석")

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
                    st.markdown("<h4 style='color:grey;'>변수 선택</h4>", unsafe_allow_html=True)
                    use_duration_column = st.checkbox("생존 '기간' 열이 이미 존재합니까?")
                    st.write(" ")
                    st.write(" ")
                    duration_column = None  # Initialize duration_column as None

                    if use_duration_column:
                        # If the duration column exists
                        st.markdown("<h5>생존 기간과 생존 상태 선택</h5>", unsafe_allow_html=True)
                        duration_column = st.selectbox("생존 기간을 나타내는 열을 선택해주세요.", options=["-- 선택 --"] + list(df.columns), index=0)
                        event_column = st.selectbox("생존 상태(1=이벤트 발생, 0=검열)를 나타내는 열을 선택해주세요.", options=["-- 선택 --"] + list(df.columns), index=0)

                        if duration_column != '-- 선택 --' and event_column != '-- 선택 --':
                            try:
                                df[duration_column] = pd.to_numeric(df[duration_column], errors='coerce')
                            except Exception as e:
                                st.error(f"'{duration_column}' 열을 numerical 형식으로 변환할 수 없습니다. 다시 확인해주세요.")

                            if not df[event_column].isin([0, 1]).all():
                                st.error("생존 상태 열(event_column)에 0과 1 이외의 값이 포함되어 있습니다. 데이터를 확인해주세요.")

                            missing_duration_count = df[duration_column].isna().sum()
                            missing_event_count = df[event_column].isna().sum()
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                            if missing_duration_count > 0 or missing_event_count > 0:
                                st.markdown(
                                    f"<p style='font-size:16px; color:red;'><strong>{missing_duration_count}개의 결측이 '{duration_column}' 열에, {missing_event_count}개의 결측이 '{event_column}' 열에 발견되었습니다.</strong></p>",
                                    unsafe_allow_html=True
                                )
                                if st.checkbox("결측된 관측을 검열로 기록하시려면 선택하세요. 미선택 시 결측 행은 분석에서 제외됩니다."):
                                    censoring_num = st.number_input("검열까지의 기간을 입력해주세요:")
                                    if censoring_num:
                                        df[duration_column] = df[duration_column].fillna(censoring_num)
                                        df[event_column] = df[event_column].fillna(0)  # Mark as censored
                            else:
                                st.markdown("<p style='font-size:16px; color:black;'><strong>결측 처리 작업 없이 분석이 가능합니다.</strong></p>", unsafe_allow_html=True)

                    else:
                        # If the duration column does not exist
                        st.markdown("<h5>생존(검열)일자 생존 상태 선택</h5>", unsafe_allow_html=True)
                        time_column = st.selectbox("생존(검열)일자를 나타내는 열을 선택해주세요.", options=["-- 선택 --"] + list(df.columns), index=0)
                        event_column = st.selectbox("생존 상태(1=이벤트 발생, 0=검열)를 나타내는 열을 선택해주세요.", options=["-- 선택 --"] + list(df.columns), index=0)

                        if time_column != '-- 선택 --' and event_column != '-- 선택 --':
                            try:
                                df[time_column] = pd.to_datetime(df[time_column], format='%Y%m%d', errors='coerce')
                            except Exception as e:
                                st.error(f"'{time_column}' 열을 datetime 형식으로 변환할 수 없습니다. 다시 확인해주세요.")

                            if not df[event_column].isin([0, 1]).all():
                                st.error("생존 상태 열(event_column)에 0과 1 이외의 값이 포함되어 있습니다. 데이터를 확인해주세요.")

                            missing_time_count = df[time_column].isna().sum()
                            missing_event_count = df[event_column].isna().sum()
                            st.divider()
                            st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                            if missing_time_count > 0 or missing_event_count > 0:
                                st.markdown(
                                    f"<p style='font-size:16px; color:red;'><strong>{missing_time_count}개의 결측이 '{time_column}' 열에, {missing_event_count}개의 결측이 '{event_column}' 열에 발견되었습니다.</strong></p>",
                                    unsafe_allow_html=True
                                )
                                if st.checkbox("결측된 관측을 검열로 기록하시려면 선택하세요. 미선택 시 결측 행은 분석에서 제외됩니다."):
                                    censoring_date = st.date_input("검열일자를 입력해주세요 (YYYY-MM-DD):")
                                    if censoring_date:
                                        censoring_date_numeric = pd.to_datetime(censoring_date, format='%Y%m%d')
                                        df[time_column] = df[time_column].fillna(censoring_date_numeric)
                                        df[event_column] = df[event_column].fillna(0)  # Mark as censored
                            else:
                                st.markdown("<p style='font-size:16px; color:black;'><strong>결측 처리 작업 없이 분석이 가능합니다.</strong></p>", unsafe_allow_html=True)

                            # Calculate durations based on the last date
                            last_date = df[time_column].max()
                            duration_column = 'duration'
                            df[duration_column] = (last_date - df[time_column]).dt.days

                            # Ensure variables are properly initialized in the session state
                            if 'analysis_started' not in st.session_state:
                                st.session_state.analysis_started = False
                            if 'km_cat_column' not in st.session_state:
                                st.session_state.km_cat_column = "-- 선택 --"

                            # UI for starting the analysis
                            if st.button("분석 시작"):
                                st.session_state.analysis_started = True  # Set the flag to indicate that analysis has started

                            # Check if the analysis has been started
                            if st.session_state.analysis_started:
                                st.divider()
                                st.header("💻 생존분석 결과", divider='rainbow')
                                st.markdown("<h4 style='color:grey;'>Kaplan-Meier Curve</h4>", unsafe_allow_html=True)

                                # Display the event DataFrame
                                st.markdown("<h6>Event Dataframe</h6>", unsafe_allow_html=True)
                                if 'time_column' in locals() and time_column in df.columns:
                                    # Display the DataFrame with time_column
                                    st.dataframe(df[[event_column, time_column, duration_column]], use_container_width=True)
                                else:
                                    # Display the DataFrame without time_column
                                    st.dataframe(df[[event_column, duration_column]], use_container_width=True)

                                # Kaplan-Meier analysis
                                durations = df[duration_column].dropna()  # Drop missing values if any
                                events = df[event_column].dropna()  # Drop missing values if any
                                kmf = KaplanMeierFitter()
                                kmf.fit(durations, event_observed=events)

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

                                # Display the event table
                                st.markdown("<h6>Event Table</h6>", unsafe_allow_html=True)
                                event_table = kmf.event_table
                                st.dataframe(event_table, use_container_width=True)

                                # Divider for categorical analysis
                                st.divider()
                                st.markdown("<h4 style='color:grey;'>Kaplan-Meier Curve with Variables</h4>", unsafe_allow_html=True)

                                # Categorical variable selection
                                km_cat_column = st.selectbox(
                                    "KM Curve를 볼 변수를 선택해주세요.",
                                    options=["-- 선택 --"] + [col for col in df.columns if df[col].nunique() < 10],
                                    index=0,
                                    key="km_cat_column"  # Use key to bind to session state
                                )

                                # Display the selected DataFrame and grouped KM curves if a variable is selected
                                if st.session_state.km_cat_column != "-- 선택 --":
                                    st.markdown("<h6>Event Dataframe</h6>", unsafe_allow_html=True)
                                    try:
                                        if 'time_column' in locals() and time_column in df.columns:
                                            st.dataframe(df[[event_column, time_column, duration_column, st.session_state.km_cat_column]], use_container_width=True)
                                        else:
                                            st.dataframe(df[[event_column, duration_column, st.session_state.km_cat_column]], use_container_width=True)
                                    except KeyError as e:
                                        st.error(f"One of the columns is not found in the DataFrame: {e}")

                                    # Plot Kaplan-Meier curves grouped by the categorical variable
                                    km_curve = go.Figure()
                                    for group in df[st.session_state.km_cat_column].dropna().unique():  # Ensure no NaN groups
                                        # Filter the DataFrame for the current group
                                        group_df = df[df[st.session_state.km_cat_column] == group]

                                        # Extract durations and events, and drop any missing values
                                        group_durations = group_df[duration_column].dropna()
                                        group_events = group_df[event_column].dropna()

                                        # Check that the durations and events are not empty before fitting the model
                                        if not group_durations.empty and not group_events.empty:
                                            # Fit the Kaplan-Meier model for the current group
                                            kmf.fit(group_durations, event_observed=group_events, label=str(group))

                                            # Use the correct label to access the survival function
                                            km_curve.add_trace(go.Scatter(
                                                x=kmf.timeline,
                                                y=kmf.survival_function_[str(group)],  # Use the label as the key
                                                mode='lines',
                                                name=f'{st.session_state.km_cat_column}: {group}'
                                            ))

                                    # Update layout for the grouped Plotly figure
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
                                        st.markdown(f"<h6>Event Table for Group: {group}</h6>", unsafe_allow_html=True)
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

                                # Function to get categorical columns, including low-cardinality numerical columns
                                def get_categorical_columns(df):
                                    categorical_columns = list(df.select_dtypes(include=['object', 'category']).columns)
                                    low_cardinality_numerical = [col for col in df.select_dtypes(exclude=['object', 'category']).columns if df[col].nunique() < 5]
                                    return categorical_columns + low_cardinality_numerical

                                # UI for variable selection
                                st.markdown("<h4 style='color:grey;'>변수 선택</h4>", unsafe_allow_html=True)
                                st.markdown(
                                    """
                                    <div class="custom-callout">
                                        <p>- 생존 분석에서는 생존 기간(duration)과 이벤트(event)를 이용해 분석합니다.</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.write(" ")

                                # Select continuous and categorical explanatory variables
                                continuous_columns = st.multiselect(
                                    "- 연속형 설명변수(X)를 선택해주세요.",
                                    df.select_dtypes(include=['float64', 'int64']).columns,
                                    key="continuous_columns_selection"
                                )


                                categorical_columns = st.multiselect(
                                    "- 범주형 설명변수(X)를 선택해주세요.",
                                    get_categorical_columns(df),
                                    key="categorical_columns_selection"
                                )

                                # Add a "선택 완료" button to confirm selections
                                if st.button('선택 완료', key='complete_button'):
                                    if continuous_columns or categorical_columns:
                                        st.session_state.continuous_columns = continuous_columns
                                        st.session_state.categorical_columns = categorical_columns
                                        st.session_state.proceed_to_preprocessing = True
                                    else:
                                        st.warning("설명변수를 한 개 이상 선택해주세요.", icon="⚠️")

                                # Check if preprocessing should proceed
                                if st.session_state.proceed_to_preprocessing:
                                    st.divider()
                                    st.markdown("<h4 style='color:grey;'>결측 파악</h4>", unsafe_allow_html=True)

                                    # Initialize dictionaries to store strategies
                                    continuous_missing_value_strategies = {}
                                    categorical_missing_value_strategies = {}

                                    # Prepare data for preprocessing
                                    X_continuous = df[st.session_state.continuous_columns].copy()
                                    X_categorical = df[st.session_state.categorical_columns].copy()

                                    # Check and display missing values for the selected variables
                                    for column in st.session_state.continuous_columns:
                                        missing_count = df[column].isna().sum()
                                        st.markdown(f"<p style='color:firebrick;'>⚠️ '{column}'에 결측치 {missing_count}개가 있습니다.</p>", unsafe_allow_html=True)
                                        # Select strategy for handling missing values
                                        if missing_count > 0:
                                            strategy = st.selectbox(
                                                f"- 선택하신 연속형 설명변수 '{column}'의 결측 처리 방법을 선택하세요:",
                                                ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체'],
                                                key=f"{column}_strategy"
                                            )
                                            if strategy != '-- 선택 --':
                                                continuous_missing_value_strategies[column] = strategy

                                    for column in st.session_state.categorical_columns:
                                        missing_count = df[column].isna().sum()
                                        st.markdown(f"<p style='color:firebrick;'>⚠️ '{column}'에 결측치 {missing_count}개가 있습니다.</p>", unsafe_allow_html=True)
                                        if missing_count > 0:
                                            # Select strategy for handling missing values
                                            strategy = st.selectbox(
                                                f"- 선택하신 범주형 설명변수 '{column}'의 결측 처리 방법을 선택하세요:",
                                                ['-- 선택 --', '결측이 존재하는 행을 제거', '해당 열의 최빈값으로 대체'],
                                                key=f"{column}_strategy"
                                            )
                                            if strategy != '-- 선택 --':
                                                categorical_missing_value_strategies[column] = strategy

                                    # Apply missing value strategies for continuous columns
                                    for column, strategy in continuous_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_continuous = X_continuous.dropna(subset=[column])
                                        elif strategy in ['해당 열의 평균값으로 대체', '해당 열의 중앙값으로 대체', '해당 열의 최빈값으로 대체']:
                                            impute_strategy = {
                                                '해당 열의 평균값으로 대체': 'mean',
                                                '해당 열의 중앙값으로 대체': 'median',
                                                '해당 열의 최빈값으로 대체': 'most_frequent'
                                            }[strategy]
                                            imputer = SimpleImputer(strategy=impute_strategy)
                                            X_continuous[[column]] = imputer.fit_transform(X_continuous[[column]])

                                    # Apply missing value strategies for categorical columns
                                    for column, strategy in categorical_missing_value_strategies.items():
                                        if strategy == '결측이 존재하는 행을 제거':
                                            X_categorical = X_categorical.dropna(subset=[column])
                                        elif strategy == '해당 열의 최빈값으로 대체':
                                            imputer = SimpleImputer(strategy='most_frequent')
                                            X_categorical[[column]] = imputer.fit_transform(X_categorical[[column]])

                                    # Synchronize index changes between X_continuous and X_categorical
                                    shared_indexes = X_continuous.index.intersection(X_categorical.index)
                                    X_continuous = X_continuous.loc[shared_indexes]
                                    X_categorical = X_categorical.loc[shared_indexes]

                                    # Combine continuous and categorical data
                                    X_combined = pd.concat([X_continuous, pd.get_dummies(X_categorical, drop_first=True)], axis=1)

                                    # Proceed to survival analysis
                                    if st.button("생존 분석 시작"):
                                        st.divider()
                                        # Prepare data for Cox model
                                        cph = CoxPHFitter()
                                        df_for_cox = pd.concat([df[[duration_column, event_column]], X_combined], axis=1).dropna()

                                        # Fit the Cox model
                                        cph.fit(df_for_cox, duration_col=duration_column, event_col=event_column)
                                        st.markdown("<h5 style='color:grey;'>Cox 모델 요약:</h5>", unsafe_allow_html=True)
                                        summary_df = cph.summary.reset_index()  # Convert summary to a DataFrame and reset the index
                                        st.dataframe(summary_df, use_container_width=True)

                                        # Extract variable names from the index of the summary DataFrame
                                        variables = summary_df.covariate.tolist()

                                        # Plot Cox model coefficients using Plotly
                                        coef = summary_df['coef']
                                        coef_lower = summary_df['coef lower 95%']
                                        coef_upper = summary_df['coef upper 95%']

                                        # Create a Plotly figure for the coefficients
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=coef,
                                            y=variables,
                                            mode='markers',
                                            name='Coefficient',
                                            error_x=dict(
                                                type='data',
                                                symmetric=False,
                                                array=coef_upper - coef,  # Upper error
                                                arrayminus=coef - coef_lower  # Lower error
                                            ),
                                            marker=dict(color='blue')
                                        ))
                                        fig.update_layout(
                                            title="Cox Model Coefficients",
                                            xaxis_title="Coefficient Value",
                                            yaxis_title="Variables",
                                            width=800,
                                            height=600
                                        )
                                        st.plotly_chart(fig)
            except ValueError as e:
                st.error("적합하지 않는 데이터를 선택하였습니다. 다시 시도해주세요.")
            except OSError as e:  # 파일 암호화 또는 해독 문제 처리
                st.error("파일이 암호화된 것 같습니다. 파일의 암호를 푼 후 다시 시도해주세요.")

    # elif page == "⛔ 오류가 발생했어요":
    #     # Mailgun API Information
    #     API_KEY = '5b177fd33abf249de3f999a97688833a-5dcb5e36-e3549260'
    #     DOMAIN_NAME = 'sandbox9b0aa132fcdb42e2a35c0642808b1f8d.mailgun.org'

    #     # Email sending function via Mailgun
    #     def send_email_via_mailgun(subject, message):
    #         try:
    #             response = requests.post(
    #                 f"https://api.mailgun.net/v3/{DOMAIN_NAME}/messages",
    #                 auth=("api", API_KEY),
    #                 data={
    #                     "from": f"Excited User <mailgun@{DOMAIN_NAME}>",  # Sender address
    #                     "to": ["hui135@snu.ac.kr"],  # Recipient address
    #                     "subject": subject,  # Email subject
    #                     "text": message  # Email body
    #                 }
    #             )
    #             return response
    #         except Exception as e:
    #             st.error(f"An error occurred: {e}")
    #             return None

    #     st.markdown(
    #     """
    #     <div style="background-color: #e9f5ff; padding: 10px; border-radius: 10px;">
    #         <h2 style="color: #000000;">⛔ 오류가 발생했어요</h2>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    #     )
    #     st.divider()
    #     st.write(" ")

    #     # Get user input
    #     st.markdown("<h4 style='color:grey;'>어떤 어려움이 있으셨나요?</h4>", unsafe_allow_html=True)
    #     user_input = st.text_area("여기에 겪고계신 어려움을 작성해주세요. 입력하신 메세지는 김희연 연구원에게 전달됩니다.", key="user_input")

    #     # Send an email when the submit button is clicked
    #     if st.button("제출", key="submit_button_1"):
    #         if user_input.strip() == "":  # Check if the input is empty
    #             st.warning("제출 전 내용을 작성해주세요.")
    #         else:
    #             response = send_email_via_mailgun("User Feedback", user_input)

    #             # If response is None, an error occurred during the request
    #             if response is None:
    #                 st.error("전송에 실패하였습니다.")
    #             else:
    #                 # Check response status code
    #                 if response.status_code == 200:
    #                     st.success("성공적으로 전송되었습니다.")
    #                 else:
    #                     st.error(f"Send failed: {response.text}")
    #                     st.write(f"Status code: {response.status_code}")

else:
    # st.markdown("<h4 style='color:grey;'>시스템 접근을 위해 로그인이 필요합니다.</h4>", unsafe_allow_html=True)
    # st.info('환영합니다!\n   강남센터 연구자 지원 이용을 위해선 좌측 사이드바에서 로그인이 필요합니다.', icon="💡")
    st.image(image_url, use_container_width=False)
    st.markdown(title_html, unsafe_allow_html=True)
    st.markdown(contact_info_html, unsafe_allow_html=True)
    st.divider()
    st.info('강남센터 연구자 지원 이용을 위해선 좌측 사이드바에서 로그인이 필요합니다.', icon="✅")
    # st.markdown(
    # """
    # <div border-radius: 10px;">
    #     <p style="font-size:20px; color: #133f91;">
    #     ☑️  강남센터 연구자 지원 이용을 위해선 좌측 사이드바에서 로그인이 필요합니다.
    #     </p>
    # </div>
    # """,
    # unsafe_allow_html=True
    # )
