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


#
# 기본 설정 초기화 한뒤 작업 디렉토리 ('work_dir'), S3 버킷 이름 ('s3bucket'), 그리고 변수 초기화
class XpopSeoul:
    def __init__(self, work_dir=".", extract_dir="."):
        self.dataset_list = None
        self.prefix = "local"
        self.work_dir = work_dir
        self.month = ""
        self.dataset_name = ""
        self.xpop_zip = ""
        self.extract_dir = extract_dir
        self.combine_dir = ""
        self.file_name = ""

    def make_combine(self):
        conn = psycopg2.connect(
            host="localhost",
            dbname="pop_platform",
            user="postgres",
            password="mim1234!",
            port=5432,
        )
        csv_files = glob.glob(self.extract_dir + "/*.csv")
        len_csv_files = len(csv_files)
        for file_index, csv in enumerate(csv_files, start=1):
            try:
                self.file_name = os.path.splitext(os.path.basename(csv))[0]
                print(self.file_name + " start")
                df = pd.read_csv(csv, na_values="*", encoding="euc_kr")
                df = self.rename_columns(df)  # for code clarity since df muted
                self.combine_datas0(df, conn, self.file_name, file_index, len_csv_files)
                print(self.file_name + " finish")
            except UnicodeDecodeError:
                df = pd.read_csv(csv, na_values="*")

        # 데이터베이스 연결 종료
        conn.close()
        print("끝!")
        return self

    # csv파일의 컬럼 이름을 사용하기 쉬운걸로 변경함. 한국어 컬럼을 영어로, 데이터의 뜻을 더 부각
    def rename_columns(self, df):
        # rename columns
        col_names_org = list(df.columns)
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

        df.columns = new_names  # df mutable affected - columns changed at this moment

        print("rename_columns start")
        print("orginal cols: ", col_names_org)
        print("changed cols: ", df.columns)
        print("rename_columns end")
        return df

    def combine_datas0(self, df, conn, file_name, file_index, len_csv_files):
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

        total_rows = len(df)
        for index, row in df.iterrows():
            print(
                f"{index+1}/{total_rows} - {file_index}/{len_csv_files} - {file_name}"
            )
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


"""
실제 동작
"""

XpopSeoul(
    work_dir="/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/",
    extract_dir="/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-202404",
).make_combine()


# test ----------------------------------------------------------------------
# ##########################################################################################
# extract_dir = "/Users/eungi/Desktop/eungi_dt/mim/livpop/src/data/local-202404"
# csv_files = glob.glob(extract_dir + "/*.csv")
# csv = csv_files[2]
# file_name = os.path.splitext(os.path.basename(csv))[0]
# df = pd.read_csv(csv, na_values="*", encoding="euc_kr")
# df
# df = rename_columns(df)  # for code clarity since df muted

# conn = psycopg2.connect(
#     host="localhost",
#     dbname="pop_platform",
#     user="postgres",
#     password="mim1234!",
#     port=5432,
# )

# combine_datas0(df, conn)
# # 데이터베이스 연결 종료
# conn.close()
# print("끝!")


# ##########################################################################################
# def rename_columns(df):
#     # rename columns
#     col_names_org = list(df.columns)
#     new_names = list()
#     for col in df.columns:
#         col = col.replace("남자", "xpop_m_")
#         col = col.replace("여자", "xpop_f_")
#         col = col.replace("세부터", "to")
#         col = col.replace("세생활인구수", "")
#         col = col.replace("세이상생활인구수", "over")
#         col = col.replace("기준일ID", "date")
#         col = col.replace('"', "")
#         col = col.replace("?", "")
#         col = col.replace("시간대구분", "hour")
#         col = col.replace("행정동코드", "adm_id")
#         col = col.replace("집계구코드", "census_id")
#         col = col.replace("총생활인구수", "xpop_total")
#         new_names.append(col)

#     df.columns = new_names  # df mutable affected - columns changed at this moment

#     print("rename_columns start")
#     print("orginal cols: ", col_names_org)
#     print("changed cols: ", df.columns)
#     print("rename_columns end")
#     return df


# def combine_datas0(df, conn):
#     # new_columns = list()
#     for col in df.columns:
#         if "xpop" in col:  # or "_m_" in col or "_f_" in col:
#             df[col] = df[col].fillna(2).astype(float)
#             # df[col] = df[col].fillna(2).astype(int)
#         elif "hour" in col:
#             df[col] = df[col].astype(int)
#         elif "date" in col:
#             pass
#         else:
#             df[col] = df[col].astype(str)

#     # new_rows = list()
#     total_rows = len(df)
#     for index, row in df.iterrows():
#         print(f"{index+1}/{total_rows}")
#         skip_next = False
#         before_key = ""
#         new_dict = dict()
#         for key in row.keys():
#             if skip_next:
#                 new_key = before_key
#                 new_key = new_key[:-1] + "9"
#                 # new_dict[new_key] = round(float(row[before_key] if row[before_key] != "nan" else 2) + float(row[key] if row[key] != "nan" else 2), 4)
#                 new_dict[new_key] = round(float(row[before_key]) + float(row[key]), 4)
#                 # new_dict[new_key] = int(row[before_key]) + int(row[key])
#                 before_key = ""
#                 skip_next = False
#                 continue

#             if "0to" in key and "9" not in key:
#                 skip_next = True
#                 before_key = key
#             else:
#                 if "xpop" in key or "_m_" in key or "_f_" in key:
#                     # new_dict[key] = int(row[key])
#                     new_dict[key] = round(float(row[key]), 4)
#                 elif "date" in key:
#                     date = row[key]
#                     # 문자열을 datetime 객체로 변환
#                     date_obj = datetime.strptime(str(date), "%Y%m%d")
#                     # datetime 객체를 원하는 형식으로 변환
#                     formatted_date = date_obj.strftime("%Y-%m-%d")

#                     new_dict["time"] = (
#                         str(formatted_date) + " " + str(row["hour"]).zfill(2) + ":00:00"
#                     )
#                 elif "hour" not in key:
#                     new_dict[key] = str(row[key])

#         try:
#             # 트랜잭션 시작
#             with conn:
#                 with conn.cursor() as cursor:
#                     # SQL 쿼리 정의
#                     query = sql.SQL(
#                         """
#                     WITH ins AS (
#                         INSERT INTO pop_info (adm_id, census_id)
#                         VALUES (%s, %s)
#                         ON CONFLICT (census_id)  -- 충돌 시
#                         DO NOTHING  -- 아무 작업도 하지 않음
#                         RETURNING pop_id  -- 삽입된 id 반환
#                     ),
#                     existing AS (
#                         SELECT pop_id
#                         FROM pop_info
#                         WHERE adm_id = %s AND census_id = %s
#                     )
#                     INSERT INTO pop_detail (pop_info_id, pop_time, pop_total, pop_m_0to9, pop_m_10to19, pop_m_20to29, pop_m_30to39, pop_m_40to49, pop_m_50to59, pop_m_60to69, pop_m_over70, pop_f_0to9, pop_f_10to19, pop_f_20to29, pop_f_30to39, pop_f_40to49, pop_f_50to59, pop_f_60to69, pop_f_over70)
#                     SELECT pop_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
#                     FROM ins
#                     UNION ALL
#                     SELECT pop_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
#                     FROM existing
#                     """
#                     )

#                     pop_time = datetime.strptime(new_dict["time"], "%Y-%m-%d %H:%M:%S")

#                     # 쿼리 실행
#                     cursor.execute(
#                         query,
#                         (
#                             new_dict["adm_id"],
#                             new_dict["census_id"],  # 새 데이터 삽입
#                             new_dict["adm_id"],
#                             new_dict["census_id"],  # 기존 데이터 확인
#                             pop_time,
#                             new_dict["xpop_total"],
#                             new_dict["xpop_m_0to9"],
#                             new_dict["xpop_m_10to19"],
#                             new_dict["xpop_m_20to29"],
#                             new_dict["xpop_m_30to39"],
#                             new_dict["xpop_m_40to49"],
#                             new_dict["xpop_m_50to59"],
#                             new_dict["xpop_m_60to69"],
#                             new_dict["xpop_m_70over"],
#                             new_dict["xpop_f_0to9"],
#                             new_dict["xpop_f_10to19"],
#                             new_dict["xpop_f_20to29"],
#                             new_dict["xpop_f_30to39"],
#                             new_dict["xpop_f_40to49"],
#                             new_dict["xpop_f_50to59"],
#                             new_dict["xpop_f_60to69"],
#                             new_dict["xpop_f_70over"],  # 삽입할 데이터
#                             pop_time,
#                             new_dict["xpop_total"],
#                             new_dict["xpop_m_0to9"],
#                             new_dict["xpop_m_10to19"],
#                             new_dict["xpop_m_20to29"],
#                             new_dict["xpop_m_30to39"],
#                             new_dict["xpop_m_40to49"],
#                             new_dict["xpop_m_50to59"],
#                             new_dict["xpop_m_60to69"],
#                             new_dict["xpop_m_70over"],
#                             new_dict["xpop_f_0to9"],
#                             new_dict["xpop_f_10to19"],
#                             new_dict["xpop_f_20to29"],
#                             new_dict["xpop_f_30to39"],
#                             new_dict["xpop_f_40to49"],
#                             new_dict["xpop_f_50to59"],
#                             new_dict["xpop_f_60to69"],
#                             new_dict["xpop_f_70over"],  # 삽입할 데이터
#                         ),
#                     )
#                     # 변경 사항 커밋
#                     conn.commit()

#         except psycopg2.Error as e:
#             print(f"Database error: {e}")


####################################################################################################
