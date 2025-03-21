import os
import shutil
import re
import glob
from datetime import datetime
import zipfile
import urllib.request
import ssl

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import ARRAY


"""
Definition 명령어 초기 세팅

(서울시 생활인구 데이터를 관리하기 위한 기능 세팅 (수정 가능))
((서울시 생활인구 데이터를 관리하는 전체적인 프로세스를 "자동화" 시키기 위해 만듬))

데이터셋 목록 가져오기
아직 처리되지 않은 데이터셋 식별
데이터 다운로드 및 압축 해제
데이터 전처리 및 변환
"""


# 기본 설정 초기화 한뒤 작업 디렉토리 ('work_dir'), S3 버킷 이름 ('s3bucket'), 그리고 변수 초기화
class XpopSeoul:
    def __init__(self, work_dir="."):
        self.dataset_list = None
        self.prefix = "local"
        self.work_dir = work_dir
        self.month = ""
        self.dataset_name = ""
        self.xpop_zip = ""
        self.extract_dir = ""
        self.combine_dir = ""
        self.file_name = ""

    # 생활인구 데이터셋의 목록을 가져옴 (웹 페이지를 스크랩하여 데이터셋의 시퀸스 번호와 날짜를 추출. 'mode'매개변수를 통해 월별 데이터만 필터링함
    def list_dataset(self, mode="monthly"):
        # file list page
        url = "https://data.seoul.go.kr/dataList/fileView.do?infId=OA-14979&srvType=F"
        page = urllib.request.urlopen(url).read()

        soup = BeautifulSoup(page, "html.parser")
        seq_numbers = list()
        for tr in soup.find_all("table", class_="dataset01")[0].find_all("tr"):
            # print(tr)
            try:
                tds = tr.find_all("td")
                filename = tds[1].text
                print(filename)
                date = re.search(r"\d+", filename).group(0)
                href = tds[5].find("a")["href"]
                seq_number = re.search(r"\d+", href).group(0)
                seq_numbers.append({"date": date, "seq_no": seq_number})
            except:
                pass
        df_seq = pd.DataFrame(seq_numbers)
        if mode == "monthly":
            df_seq = df_seq[df_seq.date.str.len() == 6]
        self.dataset_list = df_seq.reset_index(drop=True)
        self.inf_seq = soup.find("input", id="infSeq").get("value")
        print("number of available datasets: {}".format(len(df_seq)))
        return self

    # S3버킷에 저장된 데이터와 비교하여 아직 처리되지않은 데이터셋을 찾음. 이를 위해 S3버킷 내의 parquet파일목록과 현재 사용가능한 데이터셋 목록 비교함
    def find_unprocessed(self, s3bucket="xpop-seoul"):
        available = self._list_available_dataset()
        existing = self._list_parquet_in_s3(s3bucket)
        unprocessed = sorted(
            [item.replace(self.prefix + "-", "") for item in available - existing]
        )
        print("number of unprocessed datasets: {}".format(len(unprocessed)))
        return unprocessed

    # 툭정 월에 대한 작업을 설정. 내부적으로 다양한 파일경로를 설정함
    def set_month(self, month):
        self.month = month
        self.dataset_name = "{}-{}".format(self.prefix, month)
        self.xpop_zip = os.path.join(self.work_dir, "{}.zip".format(self.dataset_name))
        self.extract_dir = os.path.join(self.work_dir, self.dataset_name)
        self.combine_dir = os.path.join(self.work_dir, "combine")
        return self

    # 지정된 월에 해당하는 csv데이터셋을 다운로드하고 압축 해제함. 필요한 경우 내부 zip파일도 해제함
    def download_csvs(self):
        # Ensure the target directory exists in SageMaker environment
        work_dir_path = os.path.join("", self.work_dir)
        if not os.path.exists(work_dir_path):
            os.makedirs(work_dir_path)

        self.xpop_zip = os.path.join(work_dir_path, "{}.zip".format(self.dataset_name))
        self.extract_dir = os.path.join(work_dir_path, self.dataset_name)
        if os.path.exists(self.extract_dir):
            return self
        # Download the ZIP file
        seq_no = self._get_seq_no(self.month)
        url = (
            "http://115.84.165.224/bigfile/iot/inf/nio_download.do"
            + "?&infId=OA-14979&seq={}&infSeq={}".format(seq_no, self.inf_seq)
        )
        urllib.request.urlretrieve(url, self.xpop_zip)
        # Extract
        if os.path.exists(self.extract_dir):
            shutil.rmtree(self.extract_dir)  # Use shutil.rmtree to remove a directory
        os.makedirs(
            self.extract_dir
        )  # Use os.makedirs which can create all intermediate-level directories needed

        with zipfile.ZipFile(self.xpop_zip, "r") as zip_ref:
            zip_ref.extractall(self.extract_dir)
        # Handle nested ZIP files, if any
        zip_files = glob.glob(os.path.join(self.extract_dir, "*.zip"))
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, "r") as zip_ref_inner:
                zip_ref_inner.extractall(self.extract_dir)

        print("CSV files downloaded and extracted at {}".format(self.extract_dir))
        return self

    # 다운로드한 csv파일들을 읽고, 필요한 데이터 변환을 실행한 후, 이를 parquet파일 형식으로 저장함. 이과정에서 데이터 타입 변환, 결측치 처리 등 포함 됨
    def make_parquet(self):
        # Ensure the directory for the Parquet file exists
        parquet_dir = os.path.dirname(
            self.parquet_path
        )  # Get the directory part of the path
        if not os.path.exists(parquet_dir):
            os.makedirs(
                parquet_dir
            )  # Create the directory and any necessary parent directories
        # Proceed with your existing code to process and save the Parquet file
        csv_files = glob.glob(self.extract_dir + "/*.csv")
        dfs = list()
        for csv in csv_files:
            try:
                df = pd.read_csv(csv, na_values="*", encoding="euc_kr")
                if isinstance(df.index, pd.MultiIndex):
                    dft = df.reset_index().iloc[:, 0 : len(df.columns)]
                    dft.columns = df.columns
                    df = dft
            except UnicodeDecodeError:
                df = pd.read_csv(csv, na_values="*")
            df = self.rename_columns(df)
            dfs.append(df)
        pop = pd.concat(dfs)
        # pop = pd.concat(map(lambda p: pd.read_csv(p, na_values='*', encoding = "euc_kr"), csv_files))
        pop = pop.fillna(0)
        pop["date"] = pd.to_datetime(pop["date"], format="%Y%m%d")
        # change dtype
        for col in pop.columns:
            if "xpop" in col or "hour" in col:
                pop[col] = pop[col].astype(int)
            elif "date" in col:
                pass
            else:
                pop[col] = pop[col].astype(str)
        pop.to_parquet(self.parquet_path)
        print("parquet create at {}".format(self.parquet_path))
        return self

    def make_combine(self):
        conn = psycopg2.connect(
            host="localhost",
            dbname="pop_platform",
            user="postgres",
            password="mim1234!",
            port=5432,
        )
        # PostgreSQL 연결 설정 (SQLAlchemy 엔진 사용)
        # engine = create_engine('postgresql://postgres:mim1234!@localhost:5432/pop_platform')
        # conn = engine.connect()
        combine_dir = os.path.dirname(
            self.combine_dir
        )  # Get the directory part of the path
        if not os.path.exists(self.combine_dir):
            os.makedirs(
                self.combine_dir
            )  # Create the directory and any necessary parent directories
        csv_files = glob.glob(self.extract_dir + "/*.csv")
        combine_list = list()
        for csv in csv_files:
            try:
                self.file_name = os.path.splitext(os.path.basename(csv))[0]

                df = pd.read_csv(csv, na_values="*", encoding="euc_kr")
                self.rename_columns(df)
                # self.combine_datas(df, conn)
                self.combine_datas0(df, conn)
                # self.combine_datas2(df, conn)
                print(self.file_name + " finish")
                break
            except UnicodeDecodeError:
                df = pd.read_csv(csv, na_values="*")
            # df = self.rename_columns(df)
            # combine_list.append(df)

        # 데이터베이스 연결 종료
        conn.close()

        print("끝!")
        return self

    # csv파일의 컬럼 이름을 사용하기 쉬운걸로 변경함. 한국어 컬럼을 영어로, 데이터의 뜻을 더 부각
    def rename_columns(self, df):
        # rename columns
        col_names = list(df.columns)
        new_names = list()
        for col in df.columns:
            col = col.replace("남자", "xpop_m_")
            col = col.replace("여자", "xpop_f_")
            col = col.replace("세부터", "to")
            col = col.replace("세생활인구수", "")
            col = col.replace("세이상생활인구수", "over")
            col = col.replace("기준일ID", "date")
            col = col.replace('"', "")
            col = col.replace("?", "")
            col = col.replace("시간대구분", "hour")
            col = col.replace("행정동코드", "adm_id")
            col = col.replace("집계구코드", "census_id")
            col = col.replace("총생활인구수", "xpop_total")
            new_names.append(col)

        df.columns = new_names

        # print(df.columns)
        return df

    def combine_datas(self, df, conn):
        new_columns = list()
        sample_df = {}
        for col in df.columns:
            if "xpop" in col:  # or "_m_" in col or "_f_" in col:
                df[col] = df[col].fillna(2).astype(float)
                # df[col] = df[col].fillna(2).astype(int)
                if "total" in col:
                    sample_df[col] = df[col]
            elif "hour" in col:
                df[col] = df[col].astype(int)
            elif "date" in col:
                pass
            else:
                df[col] = df[col].astype(str)
                sample_df[col] = df[col]

        # 연령대 그룹 정의 (10살 간격)
        age_groups = {
            "0to9": ["0to9"],
            "10to19": ["10to14", "15to19"],
            "20to29": ["20to24", "25to29"],
            "30to39": ["30to34", "35to39"],
            "40to49": ["40to44", "45to49"],
            "50to59": ["50to54", "55to59"],
            "60to69": ["60to64", "65to69"],
            "70over": ["70over"],
        }

        # 연령대별 데이터 합치기
        def aggregate_age_groups(df, prefix):
            aggregated_data = {}
            for age_group, columns in age_groups.items():
                prefixed_columns = [f"{prefix}_{col}" for col in columns]
                aggregated_data[f"{prefix}_{age_group}"] = (
                    df[prefixed_columns].sum(axis=1).round(4)
                )
            return aggregated_data

        # 남성 데이터 연령대별 합산
        male_data = aggregate_age_groups(df, "xpop_m")

        # 여성 데이터 연령대별 합산
        female_data = aggregate_age_groups(df, "xpop_f")

        # 'date'과 'hour'을 합쳐서 datetime 객체 생성
        def combine_date_hour(row):
            date_str = str(row["date"])
            hour_str = str(row["hour"]).zfill(2)

            # 문자열을 datetime 객체로 변환
            date_obj = datetime.strptime(date_str, "%Y%m%d")

            # datetime 객체를 원하는 형식으로 변환
            formatted_date = date_obj.strftime("%Y-%m-%d")

            # 포맷팅된 문자열 생성
            return f"{formatted_date} {hour_str}:00:00"

        # 'date'과 'hour'을 조합하여 새로운 컬럼 'time' 생성
        sample_df["time"] = df.apply(combine_date_hour, axis=1)

        # 남성 및 여성 데이터 통합하여 sample_df에 추가
        sample_df.update(male_data)
        sample_df.update(female_data)

        # sample_df를 DataFrame으로 변환
        final_df = pd.DataFrame(sample_df)

        new_rows = list()
        for index, new_dict in final_df.iterrows():
            # print(row)
            """
            skip_next = False
            before_key = ""
            new_dict = dict()
            for key in row.keys():
                if skip_next:
                    new_key = before_key
                    new_key = new_key[:-1] + '9'
                    #new_dict[new_key] = round(float(row[before_key] if row[before_key] != "nan" else 2) + float(row[key] if row[key] != "nan" else 2), 4)
                    new_dict[new_key] = round(float(row[before_key]) + float(row[key]), 4)
                    #new_dict[new_key] = int(row[before_key]) + int(row[key])
                    before_key = ""
                    skip_next = False
                    continue

                if "0to" in key and "9" not in key:
                    skip_next = True
                    before_key = key
                else:
                    if 'xpop' in key or "_m_" in key or "_f_" in key:
                        #new_dict[key] = int(row[key])
                        new_dict[key] = round(float(row[key]), 4)
                    elif 'date' in key:
                        date = row[key]
                        # 문자열을 datetime 객체로 변환
                        date_obj = datetime.strptime(str(date), "%Y%m%d")
                        # datetime 객체를 원하는 형식으로 변환
                        formatted_date = date_obj.strftime("%Y-%m-%d")

                        new_dict['time'] = str(formatted_date) + " " + str(row['hour']).zfill(2) + ":00:00"
                    elif 'hour' not in key:
                        new_dict[key] = str(row[key])
            """

            try:
                # 트랜잭션 시작
                with conn:
                    with conn.cursor() as cursor:
                        # SQL 쿼리 정의
                        query = sql.SQL(
                            """
                        WITH ins AS (
                            INSERT INTO pop_info (adm_id, census_id)
                            VALUES (%s, %s)
                            ON CONFLICT (census_id)  -- 충돌 시
                            DO NOTHING  -- 아무 작업도 하지 않음
                            RETURNING pop_id  -- 삽입된 id 반환
                        ),
                        existing AS (
                            SELECT pop_id
                            FROM pop_info
                            WHERE adm_id = %s AND census_id = %s
                        )
                        INSERT INTO pop_detail (pop_info_id, pop_time, pop_total, pop_m, pop_f)
                        SELECT pop_id, %s, %s, %s, %s
                        FROM ins
                        UNION ALL
                        SELECT pop_id, %s, %s, %s, %s
                        FROM existing
                        """
                        )

                        pop_time = datetime.strptime(
                            new_dict["time"], "%Y-%m-%d %H:%M:%S"
                        )

                        # 쿼리 실행
                        cursor.execute(
                            query,
                            (
                                new_dict["adm_id"],
                                new_dict["census_id"],  # 새 데이터 삽입
                                new_dict["adm_id"],
                                new_dict["census_id"],  # 기존 데이터 확인
                                pop_time,
                                new_dict["xpop_total"],
                                [
                                    new_dict["xpop_m_0to9"],
                                    new_dict["xpop_m_10to19"],
                                    new_dict["xpop_m_20to29"],
                                    new_dict["xpop_m_30to39"],
                                    new_dict["xpop_m_40to49"],
                                    new_dict["xpop_m_50to59"],
                                    new_dict["xpop_m_60to69"],
                                    new_dict["xpop_m_70over"],
                                ],
                                [
                                    new_dict["xpop_f_0to9"],
                                    new_dict["xpop_f_10to19"],
                                    new_dict["xpop_f_20to29"],
                                    new_dict["xpop_f_30to39"],
                                    new_dict["xpop_f_40to49"],
                                    new_dict["xpop_f_50to59"],
                                    new_dict["xpop_f_60to69"],
                                    new_dict["xpop_f_70over"],
                                ],  # 삽입할 데이터
                                pop_time,
                                new_dict["xpop_total"],
                                [
                                    new_dict["xpop_m_0to9"],
                                    new_dict["xpop_m_10to19"],
                                    new_dict["xpop_m_20to29"],
                                    new_dict["xpop_m_30to39"],
                                    new_dict["xpop_m_40to49"],
                                    new_dict["xpop_m_50to59"],
                                    new_dict["xpop_m_60to69"],
                                    new_dict["xpop_m_70over"],
                                ],
                                [
                                    new_dict["xpop_f_0to9"],
                                    new_dict["xpop_f_10to19"],
                                    new_dict["xpop_f_20to29"],
                                    new_dict["xpop_f_30to39"],
                                    new_dict["xpop_f_40to49"],
                                    new_dict["xpop_f_50to59"],
                                    new_dict["xpop_f_60to69"],
                                    new_dict["xpop_f_70over"],
                                ],  # 삽입할 데이터
                            ),
                        )
                        # 변경 사항 커밋

            except psycopg2.Error as e:
                print(f"Database error: {e}")

            # new_rows.append(new_dict)

        conn.commit()

        # print(new_rows)
        # 데이터프레임으로 변환
        """df2 = pd.DataFrame(new_rows)

        # CSV 파일로 내보내기
        df2.to_csv(os.path.join(self.combine_dir, "{}_combine2.csv".format(self.file_name)), index=False)

        return df2"""

    def combine_datas0(self, df, conn):
        new_columns = list()
        for col in df.columns:
            if "xpop" in col:  # or "_m_" in col or "_f_" in col:
                df[col] = df[col].fillna(2).astype(float)
                # df[col] = df[col].fillna(2).astype(int)
            elif "hour" in col:
                df[col] = df[col].astype(int)
            elif "date" in col:
                pass
            else:
                df[col] = df[col].astype(str)

        new_rows = list()
        for index, row in df.iterrows():
            # print(row)
            skip_next = False
            before_key = ""
            new_dict = dict()
            for key in row.keys():
                if skip_next:
                    new_key = before_key
                    new_key = new_key[:-1] + "9"
                    # new_dict[new_key] = round(float(row[before_key] if row[before_key] != "nan" else 2) + float(row[key] if row[key] != "nan" else 2), 4)
                    new_dict[new_key] = round(
                        float(row[before_key]) + float(row[key]), 4
                    )
                    # new_dict[new_key] = int(row[before_key]) + int(row[key])
                    before_key = ""
                    skip_next = False
                    continue

                if "0to" in key and "9" not in key:
                    skip_next = True
                    before_key = key
                else:
                    if "xpop" in key or "_m_" in key or "_f_" in key:
                        # new_dict[key] = int(row[key])
                        new_dict[key] = round(float(row[key]), 4)
                    elif "date" in key:
                        date = row[key]
                        # 문자열을 datetime 객체로 변환
                        date_obj = datetime.strptime(str(date), "%Y%m%d")
                        # datetime 객체를 원하는 형식으로 변환
                        formatted_date = date_obj.strftime("%Y-%m-%d")

                        new_dict["time"] = (
                            str(formatted_date)
                            + " "
                            + str(row["hour"]).zfill(2)
                            + ":00:00"
                        )
                    elif "hour" not in key:
                        new_dict[key] = str(row[key])

            try:
                # 트랜잭션 시작
                with conn:
                    with conn.cursor() as cursor:
                        # SQL 쿼리 정의
                        query = sql.SQL(
                            """
                        WITH ins AS (
                            INSERT INTO pop_info (adm_id, census_id)
                            VALUES (%s, %s)
                            ON CONFLICT (census_id)  -- 충돌 시
                            DO NOTHING  -- 아무 작업도 하지 않음
                            RETURNING pop_id  -- 삽입된 id 반환
                        ),
                        existing AS (
                            SELECT pop_id
                            FROM pop_info
                            WHERE adm_id = %s AND census_id = %s
                        )
                        INSERT INTO pop_detail (pop_info_id, pop_time, pop_total, pop_m_0to9, pop_m_10to19, pop_m_20to29, pop_m_30to39, pop_m_40to49, pop_m_50to59, pop_m_60to69, pop_m_over70, pop_f_0to9, pop_f_10to19, pop_f_20to29, pop_f_30to39, pop_f_40to49, pop_f_50to59, pop_f_60to69, pop_f_over70)
                        SELECT pop_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        FROM ins
                        UNION ALL
                        SELECT pop_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        FROM existing
                        """
                        )

                        pop_time = datetime.strptime(
                            new_dict["time"], "%Y-%m-%d %H:%M:%S"
                        )

                        # 쿼리 실행
                        cursor.execute(
                            query,
                            (
                                new_dict["adm_id"],
                                new_dict["census_id"],  # 새 데이터 삽입
                                new_dict["adm_id"],
                                new_dict["census_id"],  # 기존 데이터 확인
                                pop_time,
                                new_dict["xpop_total"],
                                new_dict["xpop_m_0to9"],
                                new_dict["xpop_m_10to19"],
                                new_dict["xpop_m_20to29"],
                                new_dict["xpop_m_30to39"],
                                new_dict["xpop_m_40to49"],
                                new_dict["xpop_m_50to59"],
                                new_dict["xpop_m_60to69"],
                                new_dict["xpop_m_70over"],
                                new_dict["xpop_f_0to9"],
                                new_dict["xpop_f_10to19"],
                                new_dict["xpop_f_20to29"],
                                new_dict["xpop_f_30to39"],
                                new_dict["xpop_f_40to49"],
                                new_dict["xpop_f_50to59"],
                                new_dict["xpop_f_60to69"],
                                new_dict["xpop_f_70over"],  # 삽입할 데이터
                                pop_time,
                                new_dict["xpop_total"],
                                new_dict["xpop_m_0to9"],
                                new_dict["xpop_m_10to19"],
                                new_dict["xpop_m_20to29"],
                                new_dict["xpop_m_30to39"],
                                new_dict["xpop_m_40to49"],
                                new_dict["xpop_m_50to59"],
                                new_dict["xpop_m_60to69"],
                                new_dict["xpop_m_70over"],
                                new_dict["xpop_f_0to9"],
                                new_dict["xpop_f_10to19"],
                                new_dict["xpop_f_20to29"],
                                new_dict["xpop_f_30to39"],
                                new_dict["xpop_f_40to49"],
                                new_dict["xpop_f_50to59"],
                                new_dict["xpop_f_60to69"],
                                new_dict["xpop_f_70over"],  # 삽입할 데이터
                            ),
                        )
                        # 변경 사항 커밋
                        conn.commit()

            except psycopg2.Error as e:
                print(f"Database error: {e}")

            # new_rows.append(new_dict)

        """
        #print(new_rows)
        # 데이터프레임으로 변환
        df2 = pd.DataFrame(new_rows)

        # CSV 파일로 내보내기
        df2.to_csv(os.path.join(self.combine_dir, "{}_combine.csv".format(self.file_name)), index=False)

        return df2"""

    def combine_datas2(self, df, conn):
        for col in df.columns:
            if "xpop" in col:  # or "_m_" in col or "_f_" in col:
                df[col] = df[col].fillna(2).astype(float)
            elif "hour" in col:
                df[col] = df[col].astype(int)
            elif "date" in col:
                pass
            else:
                df[col] = df[col].astype(str)

        # 연령대 그룹 정의 (10살 간격)
        age_groups = {
            "0to9": ["xpop_m_0to9"],
            "10to19": ["xpop_m_10to14", "xpop_m_15to19"],
            "20to29": ["xpop_m_20to24", "xpop_m_25to29"],
            "30to39": ["xpop_m_30to34", "xpop_m_35to39"],
            "40to49": ["xpop_m_40to44", "xpop_m_45to49"],
            "50to59": ["xpop_m_50to54", "xpop_m_55to59"],
            "60to69": ["xpop_m_60to64", "xpop_m_65to69"],
            "70over": ["xpop_m_70over"],
        }

        # 연령대별 데이터 합치기
        def aggregate_age_groups(df, prefix):
            aggregated_data = {}
            for age_group, columns in age_groups.items():
                aggregated_data[f"{prefix}_{age_group}"] = df[columns].sum(axis=1)
            return aggregated_data

        # 남성 및 여성 데이터 합치기
        df_m = pd.DataFrame(aggregate_age_groups(df, "pop_m"))
        df_f = pd.DataFrame(aggregate_age_groups(df, "pop_f"))

        # 소수점 4자리로 반올림
        df_m = df_m.round(4)
        df_f = df_f.round(4)

        # 배열 형식으로 변환
        def convert_to_array(df, columns):
            return df[columns].values.tolist()

        # date와 hour 열을 문자열로 변환하고 결합
        def combine_date_hour(date, hour):
            # print(date)
            # date를 문자열로 변환하고, datetime 객체로 변환
            date_obj = datetime.strptime(str(date), "%Y%m%d")
            # hour를 2자리 문자열로 변환
            hour_str = str(hour).zfill(2)
            # 결합하여 datetime 문자열 생성
            datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {hour_str}:00:00"
            return datetime_str

        # 연령대별 데이터를 배열로 변환
        df_m_arrays = df_m.apply(
            lambda row: convert_to_array(pd.DataFrame([row]), df_m.columns), axis=1
        )
        df_f_arrays = df_f.apply(
            lambda row: convert_to_array(pd.DataFrame([row]), df_f.columns), axis=1
        )

        # date와 hour 열을 결합하여 pop_time 열 생성
        df["pop_time"] = df.apply(
            lambda row: combine_date_hour(row["date"], row["hour"]), axis=1
        )

        # 원본 데이터에서 기본 정보 추출
        df_pop_info = df[["adm_id", "census_id"]]

        # 원본 데이터에서 pop_detail에 필요한 정보 추출
        df_pop_detail = df[["adm_id", "census_id", "pop_time", "xpop_total"]].copy()
        df_pop_detail = pd.concat(
            [
                df_pop_detail.reset_index(drop=True),
                pd.DataFrame({"pop_m": df_m_arrays, "pop_f": df_f_arrays}),
            ],
            axis=1,
        )

        # 데이터베이스에서 기존 데이터를 읽어옴
        existing_ids = pd.read_sql(
            """
            SELECT adm_id, census_id FROM pop_info
        """,
            conn,
        )

        existing_ids_set = set(tuple(x) for x in existing_ids.to_numpy())

        # 데이터프레임에서 새로 삽입할 데이터만 필터링
        new_data = df_pop_info[
            ~df_pop_info.apply(
                lambda row: (row["adm_id"], row["census_id"]) in existing_ids_set,
                axis=1,
            )
        ]

        # 새 데이터 삽입
        if not new_data.empty:
            new_data.to_sql("pop_info", conn, if_exists="append", index=False)

        # 삽입된 pop_info_id 얻기
        query = """
        SELECT adm_id, census_id, pop_id
        FROM pop_info
        WHERE (adm_id, census_id) IN ({values})
        """

        # 값을 튜플 형태로 변환
        values = ", ".join(
            [f"({row.adm_id}, {row.census_id})" for _, row in df_pop_info.iterrows()]
        )
        df_pop_ids = pd.read_sql(query.format(values=values), conn)

        # pop_detail에 pop_info_id 추가
        df_pop_detail = df_pop_detail.merge(
            df_pop_ids, on=["adm_id", "census_id"], how="left"
        )
        df_pop_detail.rename(columns={"pop_id": "pop_info_id"}, inplace=True)

        # 데이터 프레임에서 필요한 컬럼만 추출
        df_pop_detail = df_pop_detail[
            ["pop_time", "pop_total", "pop_m", "pop_f", "pop_info_id"]
        ]

        # pop_detail 테이블에 데이터 삽입
        df_pop_detail.to_sql(
            "pop_detail",
            conn,
            if_exists="append",
            index=False,
            dtype={"pop_m": ARRAY(float), "pop_f": ARRAY(float)},
        )

    # 작업에 사용된 임시 디렉토리를 정리
    def clean_up(self):
        if os.path.exists(self.xpop_zip):
            os.remove(self.xpop_zip)
        # shutil.rmtree(self.extract_dir)
        return self

    # 내부적으로 사용되는 방법으로, 지정된 월에 해당하는 데이터셋의 시퀸스 번호를 찾음
    def _get_seq_no(self, month):
        return self.dataset_list[self.dataset_list.date == month]["seq_no"].values[0]

    # 사용 가능한 데이터셋의 목록을 보여줌. 이건 'list_dataset' definition'에 이용됨
    def _list_available_dataset(self):
        return set(self.prefix + "-" + self.dataset_list["date"])


"""
실제 동작
"""

XpopSeoul(
    work_dir="/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/"
).list_dataset().set_month("202305").download_csvs().clean_up()

# XpopSeoul(work_dir="./data/").list_dataset().set_month(
#     "202403"
# ).download_csvs().make_combine().clean_up()
