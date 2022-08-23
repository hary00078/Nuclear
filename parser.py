import pickle
import gzip
import numpy as np
import os
from tqdm import tqdm
from re import IGNORECASE
import pathlib
import math
import random
import struct 
import pandas as pd
from scipy.signal import find_peaks

##### Data Parsing 클래스 #####
class DatParser():

## input_path = 데이터 저장위치(FULL)
## filename = 파일명(PNS No)
    
    def __init__(self, input_path, filename):  
        self.channel_info = dict()
        self.channel_names = []
        self.channel_datas = dict()
        self.input_path = input_path
        self.filename = filename


    def set_file(self):
#         print('input_path', input_path)
        with open(self.input_path, 'rb') as f:
            self.data = f.read()
            # self.data_2 = f.read()
        _tmp = self.data.hex()
        self.hex = [_tmp[i:i+2] for i in range(0, len(_tmp), 2)]


    def byte_to_str(self, byte_array):
        if isinstance(byte_array, int):
            return chr(byte_array)
        elif isinstance(byte_array, bytes):
            return ''.join([chr(byte) for byte in byte_array])


    def byte_to_int(self, byte_array):
        if isinstance(byte_array, int):
            return byte_array
        elif isinstance(byte_array, bytes):
            #byte_array += b'\x00' * (4-len(byte_array))
            try:
                if len(byte_array) == 2:
                    return struct.unpack("<h", byte_array)[0]
                elif len(byte_array) == 4:
                    return struct.unpack("<L", byte_array)[0]

            except Exception as e :
                print('ERROR', byte_array)
                return -1

    def byte_to_float(self, byte_array):
        if isinstance(byte_array, int):
            return byte_array
        elif isinstance(byte_array, bytes):
            byte_array = b'\x00' * 2 + byte_array
            try:
                if len(byte_array) == 2:
                    return struct.unpack("<e", byte_array)[0]
                elif len(byte_array) == 4:
                    return struct.unpack("<f", byte_array)[0]

            except Exception as e :
                print('ERROR', byte_array)
                return -1

    def get_channel_names(self):
        data = self.data
        start = data.find(b'Digital Output')
        start += 0xB3
        ch_name_first = data[start:start+10].decode().replace('\x00','').replace('R_','')

        for i in range(8):
            ch_name = data[start:start+10].decode().replace('\x00','').replace('R_','')
            # Start : Start + 9 -> 7로 수정.
            if ch_name[-1] == '0' :
                self.channel_names.append(data[start:start+10].decode().replace('\x00','').replace('R_',''))
                start += 0x90

        if len(self.channel_names) < 7 :
            print('●●●●●●●●●●●채널개수가 비정상 입니다.●●●●●●●●●●●')

        print('self.channel_name(정상): ', self.channel_names, 'self.channel_name(len): ', len(self.channel_names))


        # 일부 파일의 경우 Digital Output 이후에 다른 값이 들어 있어서 채널명 인식에 방해가 되는 경우가 있음
        # 해결하기위해 main 채널 인식 이후 sub 채널들을 자동으로 가져 오도록 변경
        # 문제가 지속적으로 발생 시 채널명을 필요 없이 동작하는 코드로 수정가능

    #         print('channel_names', self.channel_names)            

    
    def get_setting(self):
        data = self.data
        channel_info = self.channel_info

        start = data.find(b"REM Configuration file.")
        end = data.find(b"GeneralConfig")
        str1 = data[start:end].decode('utf-8', 'ignore')
        key = ''
        value = 0

        for line in str1.split('\n'):
            if 'DEFINE channel' in line:
                key = line.split('"')[1]
            if 'SET valrotation' in line:
                value = float(line.split('TO')[1].strip())
            if key and value:
                channel_info[key] = value
                key = ''
                value = 0
        print('self.channel_info', self.channel_info)
        
    def rotate_xy_series(self, x_series, y_series, degree):
        radians = math.radians(degree)
        xx = x_series * math.cos(radians) + y_series * math.sin(radians)
        yy = -x_series * math.sin(radians) + y_series * math.cos(radians)
        return xx, yy
    
    def parse(self):
        str1 = b'Version : '
        shift_offset = 0
        end_offset = 0
        self.version = self.data.index(str1)
        num2_list = []
        test_list = []

        version = self.byte_to_str(self.data[self.version+0x0a : self.version+0x0a+5])
        self.parsing_data = pd.DataFrame()
        start_offset = self.data.index(str1)
#         print('version', version)
        end_offset = len(self.data) - 130000
      
        # 고압급수가열기 Tube end 부분 짤림현상으로 인해 값 수정
        # 각 Version 별 수정필요 (6.0R8, 6.2Q5 : 140000개 만족 6.0R4:140000 이상함.)


        if len(self.channel_names) == 8 :
            if version == '6.0R8' :
                shift_offset = 0x10
                length = int(((end_offset - start_offset + shift_offset)//36)*18)
                sample_len = int((length//18))
                reshape_num = 18
              
            elif version  == '6.0R4' :
                shift_offset = 0x10      
                length = int(((end_offset - start_offset+shift_offset)//32)*16)
                sample_len = int((length//16))
                reshape_num = 16

            elif version == '6.2Q5' :
                shift_offset = 0x12  
                length = int(((end_offset - start_offset+ shift_offset)//32)*16)
                sample_len = int((length//16))
                reshape_num = 16

            start_offset = start_offset + shift_offset

            for i in range(0, length):
                num2 = self.byte_to_int(self.data[start_offset+(2*i):start_offset+2+(2*i)])
                num2_list.append(num2)
                    
            num2_list = np.array(num2_list)
            num2_final_list = num2_list.reshape(-1,reshape_num)


            df = pd.DataFrame(num2_final_list)

            if version == '6.0R8' :
                df = df.drop(df.columns[16], axis=1)
                df = df.drop(df.columns[16], axis=1)


            for i in range(0,16,2) :
                dict_name = self.channel_names[(i//2)]
                x_series = pd.Series(df[i])
                y_series = pd.Series(df[i+1])
                x_series, y_series = self.rotate_xy_series(x_series, y_series, -self.channel_info[dict_name])
                self.channel_datas[f'{dict_name}_x'] = x_series
                self.channel_datas[f'{dict_name}_y'] = y_series
                x = pd.DataFrame(self.channel_datas[f'{dict_name}_x'])
                y = pd.DataFrame(self.channel_datas[f'{dict_name}_y'])
                x.rename(columns={ 0 : f'{dict_name}_x'}, inplace=True)
                y.rename(columns={ 0 : f'{dict_name}_y'}, inplace=True)
                self.parsing_data = pd.concat([self.parsing_data, x, y], axis=1)

            self.parsing_data.columns = ['CH1X','CH1Y','CH3X','CH3Y','CH5X','CH5Y','CH7X','CH7Y','CH2X','CH2Y','CH4X','CH4Y','CH6X','CH6Y','CH8X','CH8Y']

        if len(self.channel_names) == 7 :
            print('self.channel_name =7개 비정상입니다.')
#             shift_offset = 10000
#             length = int(((end_offset - start_offset)//28)*14)
#             sample_len = int((length//14))
#             reshape_num = 14

#             shift_offset = 0x12  
#             length = int(((end_offset - start_offset+ shift_offset)//28)*14)
#             sample_len = int((length//14))
#             reshape_num = 14

#             start_offset = start_offset + shift_offset

#             for i in range(0, length):
#               num2 = self.byte_to_int(self.data[start_offset+(2*i):start_offset+2+(2*i)])
#               num2_list.append(num2)
                    
#             num2_list = np.array(num2_list)
#             num2_final_list = num2_list.reshape(-1,reshape_num)

#             df = pd.DataFrame(num2_final_list)

#             for i in range(0,14,2) :
#                 dict_name = self.channel_names[(i//2)]
#                 x_series = pd.Series(df[i])
#                 y_series = pd.Series(df[i+1])
#                 x_series, y_series = rotate_xy_series(x_series, y_series, -self.channel_info[dict_name])
#                 self.channel_datas[f'{dict_name}_x'] = x_series
#                 self.channel_datas[f'{dict_name}_y'] = y_series
#                 x = pd.DataFrame(self.channel_datas[f'{dict_name}_x'])
#                 y = pd.DataFrame(self.channel_datas[f'{dict_name}_y'])
#                 x.rename(columns={ 0 : f'{dict_name}_x'}, inplace=True)
#                 y.rename(columns={ 0 : f'{dict_name}_y'}, inplace=True)
#                 final_data = pd.concat([final_data, x, y], axis=1)

#             final_data.columns = ['CH1X','CH1Y','CH3X','CH3Y','CH5X','CH5Y','CH7X','CH7Y','CH2X','CH2Y','CH4X','CH4Y','CH6X','CH6Y']

        # final_data.to_pickle(f'/content/drive/MyDrive/Parsing Tool Check/2022.01.28_parsing/result/{self.name}.pkl.gz', compression='gzip')
 

## input_path = 데이터 저장위치(FULL)
## filename = 파일명(PNS No)
#### 라벨링 툴 수행을 위해 필수 데이터 Load master_df
# master_df = pd.read_pickle("./CD_Master_rev2.pkl")
# master_df = pd.read_pickle(r"C:\Users\user\Desktop\DeepAI\CD_Master_rev2.pkl.gz", compression='gzip')

### 라벨링 툴 Class ###
class Labeling_Tool:
    def __init__(self, output_path):
        self.df = final_data # 파싱된 데이터 : 데이터프레임 
        self.df_7Y = self.df['CH7Y'].to_numpy() # 'CH7Y 데이터 : numpy(구조물 찾는 데이터)'
        self.df_5Y = self.df['CH5Y'].to_numpy() # 'CH5Y 데이터 : numpy(구조물 찾는 데이터)'
        self.df_1Y = self.df['CH1Y'] # 'CH1Y 데이터 : '

        self.output_path = output_path

        #마스터 파일의 데이터 가져와 결함영역 찾기
        self.TSP_num = master_df['TSP'][master_df.index[master_df['PNS No']==filename]]  #결함의 TSP 위치
        self.Dgree = master_df['Deg.'][master_df.index[master_df['PNS No']==filename]]   #결함의 각도
        self.Evaluation = master_df['Eval.'][master_df.index[master_df['PNS No']==filename]]   #결함의 종류
        self.TSP_count = int(master_df['TSP_count'][master_df.index[master_df['PNS No']==filename]])   #TSP 전체 수량
     
        #2차 결함영역 인덱스 저장하는 변수
        self.second_defect_start = 0 # 결함 시작점
        self.second_defect_end = 0 # 결함 끝점
        self.second_defect_length = 0 # 결함구간 길이
        self.required_length = 100  # 원하는 결함 길이
            
        self.dgree_CH1 = 0
        self.error_code = []
        self.result_data = []
        self.find_cal_degree = 0

#         print('TSP 실제개수 :', self.TSP_count, '           결함 위치(TSP No.) :', self.TSP_num.values)
#         print('결함종류 : ', self.Evaluation.values)



    # TEI & TEO 양끝 및 중심 찾기    
    def find_TEITEO_Candidate(self):
        print('\n')
        print('\033[1m' + '\033[95m' + '1. find_TEITEO_Candidate' + '\033[0m')

        x_std = np.std(self.df_7Y)
        TEI_TEO_std = self.df_7Y.max() * 0.3

        # High Peaks 찾기
        high_peaks, _ = find_peaks(self.df_7Y, width=2, distance=1000, height=TEI_TEO_std)
        high_peaks = high_peaks.tolist()
        self.high_peaks = high_peaks
        print('   - TEI, TEO high 후보 =', self.high_peaks)
        # Low Peaks 찾기
        low_peaks, _ = find_peaks(-self.df_7Y, width=2, distance=1000, height=TEI_TEO_std)
        low_peaks = low_peaks.tolist()
        self.low_peaks = low_peaks
        print('   - TEI, TEO low 후보 =',self.low_peaks)

        if len(self.high_peaks) > len(self.low_peaks) :
            self.low_peaks = np.asarray(self.high_peaks) + 150
            self.low_peaks.tolist()
            print('   - TEI, TEO :' , self.high_peaks, self.low_peaks)

        if len(self.high_peaks) < len(self.low_peaks) :
            self.high_peaks = np.asarray(self.low_peaks) - 150
            self.high_peaks.tolist()
#             print('TEI, TEO :' , self.high_peaks, self.low_peaks)

        if len(self.high_peaks) == len(self.low_peaks) == 1 :
            print('   - TEI, TEO 갯수 1개 확인됨, 인식실패')
            self.error_code.append('TEI, TEO 갯수 1개 확인됨')

           
            
    # TEI, TEO 찾기 (find_longest 참고)
    # 가장 긴 구간이 실제 DATA가 존재하는 구간
    def find_TEI_TEO_high_low(self) :
        print('\033[1m' + '\033[95m' + '2. find_TEI_TEO_high_low' + '\033[0m')
        length = 0
        teo_high1 = 0
        tei_high1 = 0
        teo_low1 = 0
        tei_low1 = 0

        for i in range(0, len(self.high_peaks)-1) :
            if self.high_peaks[i+1] - self.high_peaks[i] > length :
                length = self.high_peaks[i+1] - self.high_peaks[i]
                teo_high1 = self.high_peaks[i+1]
                tei_high1 = self.high_peaks[i]  

        self.TEI_high = tei_high1
        self.TEO_high = teo_high1

        length = 0
        for i in range(0, len(self.low_peaks)-1) :
            if self.low_peaks[i+1] - self.low_peaks[i] > length :
                length = self.low_peaks[i+1] - self.low_peaks[i]
                teo_low1 = self.low_peaks[i+1]
                tei_low1 = self.low_peaks[i]  
        self.TEI_low = tei_low1
        self.TEO_low = teo_low1
        print(f'   - TEI_high ~ TEI_low : {self.TEI_high} ~ {self.TEI_low}')
        print(f'   - TEO_high ~ TEO_low : {self.TEO_high} ~ {self.TEO_low}')

        
        
    # just : 딱 맞게 자르다
    def find_just_TEI(self):
        print('\033[1m' + '\033[95m' + '3. find_just_TEI' + '\033[0m')

        df4 = self.df_1Y[self.TEI_low+30: self.TEI_low+1500]
        peaks, properties = find_peaks(df4, width=2, distance=200)
        if len(peaks) > 2 :
            print('   - just_TEI 검출 실패', peaks)
            peaks = peaks[0]
            
        just_TEI_peaks = peaks + self.TEI_low+30
        self.just_TEI = just_TEI_peaks+100
        print('   - just_TEI =',  self.just_TEI )

    def find_just_TEO(self):
        print('\033[1m' + '\033[95m' + '4. find_just_TEO' + '\033[0m')
        df5 = self.df_1Y[self.TEO_high-1500: self.TEO_high-30]
        
        peaks, properties = find_peaks(-df5, width=2, distance=200)
        if len(peaks) > 2 :
            print('   - just_TEO 검출 실패', peaks)
            peaks = peaks[-1]
            
        just_TEO_peaks = peaks + self.TEO_high-1500
        self.just_TEO = just_TEO_peaks-100
        print('   - just_TEO =',  self.just_TEO )

        
        
    # TSP 양끝 및 중심 찾기, df_7Y 채널 활용
    def find_TSP(self):
        print('\033[1m' + '\033[95m' + '5. find_TSP' + '\033[0m')
        self.real_data = final_data[self.just_TEI : self.just_TEO]
        x_std = np.std(self.df_7Y)
#         print(self.real_data['CH7Y'].max())
      
        try :
            print('   - 1차 TSP인식 시작(range800)')
            for weight in range(0, 800, 1):
                peaks, properties = find_peaks(self.df_7Y, width=5, distance=200, height=(x_std*(8-weight/100)*2, x_std*8))
                self.high_peak_list = peaks
                self.high_peak_list = self.high_peak_list.tolist()
                
                for i in range(0,len(self.high_peak_list)) :
                    
                    if len(self.high_peak_list) > 6:
                        while self.high_peak_list[0] < self.just_TEI :
                            del self.high_peak_list[0]
                        # print('self.high_peak_list : ', self.high_peak_list)    
                    for j in range(0, len(self.high_peak_list)) :
                        if len(self.high_peak_list) > 6:
                            while self.high_peak_list[-1] > self.just_TEO :
                                del self.high_peak_list[-1]
                    if len(self.high_peak_list) == int(self.TSP_count) :
                        break
                if len(self.high_peak_list) == int(self.TSP_count):
                    break
#             print('   - len(self.high_peak_list) : ', len(self.high_peak_list))
#             print('   - self.TSP_count : ', self.TSP_count)
#             print('   - high_weight : ', weight, (8-weight/100))

            for weight in range(0, 800, 1):
                peaks1, properties = find_peaks(-self.df_7Y, width=2, distance=100, height=(x_std*(8-weight/100)*2, x_std*8))
                self.low_peak_list = peaks1
                self.low_peak_list = self.low_peak_list.tolist()

                for i in range(0,len(self.low_peak_list)) :
                    if len(self.low_peak_list) > 6:
                        while self.low_peak_list[0] < self.just_TEI :
                            del self.low_peak_list[0]
                    for j in range(0, len(self.low_peak_list)) :
                        if len(self.low_peak_list) > 6:
                            while self.low_peak_list[-1] > self.just_TEO :
                                del self.low_peak_list[-1]
                    if len(self.low_peak_list) == int(self.TSP_count) :
                        break
                if len(self.low_peak_list) == int(self.TSP_count):
                    break
#             print('   - low_weight : ', weight, (8-weight/100))
            print('   - 1차 TSP인식 성공')
             # TSP Center 찾기
            self.TSP_Center_list = np.round((np.asarray(self.high_peak_list) + np.asarray(self.low_peak_list))/2).tolist()        

            
            
    # find_TSP로 찾지 못할 경우 2차적으로 더 weight를 세밀하게 하여 TSP 양끝 및 중심 찾기, df_7Y 채널 활용
        except :
            print('   - 2차 TSP인식 시작(range8000)')
            for weight in range(0, 8000, 1):
                peaks, properties = find_peaks(self.df_7Y, width=2, distance=100, height=(x_std*(8-weight/1000)*2,x_std*8))
                self.high_peak_list = peaks
                self.high_peak_list = self.high_peak_list.tolist()

                for i in range(0,len(self.high_peak_list)) :
                    if len(self.high_peak_list) > 6:
                        while self.high_peak_list[0] < self.just_TEI :
                            del self.high_peak_list[0]  
                    for j in range(0, len(self.high_peak_list)) :
                        if len(self.high_peak_list) > 6:
                            while self.high_peak_list[-1] > self.just_TEO :
                                del self.high_peak_list[-1]
                    if len(self.high_peak_list) == int(self.TSP_count) :
                        break
                if len(self.high_peak_list) == int(self.TSP_count):
                    break

#             print('x_std : ', x_std)       
            for weight in range(0, 8000, 1):
                peaks1, properties = find_peaks(-self.df_7Y, width=2, distance=100, height=(x_std*(8-weight/1000)*2,x_std*8))
                self.low_peak_list = peaks1
                self.low_peak_list = self.low_peak_list.tolist()

                for i in range(0,len(self.low_peak_list)) :
                    if len(self.low_peak_list) > 6:
                        while self.low_peak_list[0] < self.just_TEI :
                            del self.low_peak_list[0] 
                    for j in range(0, len(self.low_peak_list)) :
                        if len(self.low_peak_list) > 6:
                            while self.low_peak_list[-1] > self.just_TEO :
                                del self.low_peak_list[-1]
                    if len(self.low_peak_list) == int(self.TSP_count) :
                        break
                if len(self.low_peak_list) == int(self.TSP_count):
                    break
#             print(self.high_peak_list)
#             print(self.low_peak_list)
            self.TSP_Center_list = np.round((np.asarray(self.high_peak_list) + np.asarray(self.low_peak_list))/2).tolist()
            print('   - 2차 TSP인식 성공')
        
        # TSP 다 찾은 이후에 TSP 정상적으로 찾았는지 여부 확인과정. high와 low의 거리가 300 이상이면 Error다                  
        for i in range(0,len(self.high_peak_list)) :
            if np.asarray(self.high_peak_list[i]) - np.asarray(self.low_peak_list[i]) > 300 :
                self.error_code.append('   - 7Y채널 TSP 인식실패')
            if np.asarray(self.low_peak_list[i]) - np.asarray(self.high_peak_list[i]) > 300 :
                self.error_code.append('   - 7Y채널 TSP 인식실패')
                
        print('   - find_TSP 성공')
        print(f'   - self.high_peak_list : {self.high_peak_list}')
        print(f'   - self.low_peak_list : {self.low_peak_list}')
    # find_TSP와 거의 동일한 코드, 채널 df_5Y활용
    # high_peak_list_2 와 low_peak_list_2가 만들어짐, Evaluation group2, get_TSP에서 사용됨
    # 결과물 self.TSP_Center_list_2 은 생성되지만 실제 사용처는 없음
    
    def find_TSP_2(self):
        print('\033[1m' + '\033[95m' + '6. find_TSP_2' + '\033[0m')
        x_std = np.std(self.df_5Y)

        try:
            for weight in range(0, 800, 1):
                peaks, properties = find_peaks(self.df_5Y, width=5, distance=200, height=(x_std*(8-weight/100),x_std*8))
                self.high_peak_list_2 = peaks
                self.high_peak_list_2 = self.high_peak_list_2.tolist()

                for i in range(0,len(self.high_peak_list_2)) :
                    if len(self.high_peak_list_2) > 6:
                        while self.high_peak_list_2[0] < self.just_TEI :
                            del self.high_peak_list_2[0] 
                    for j in range(0, len(self.high_peak_list_2)) :
                        if len(self.high_peak_list_2) > 6:
                            while self.high_peak_list_2[-1] > self.just_TEO :
                                del self.high_peak_list_2[-1]
                    if len(self.high_peak_list_2) == int(self.TSP_count) :
                        break
                if len(self.high_peak_list_2) == int(self.TSP_count):
                    break

            for weight in range(0, 800, 1):
                peaks1, properties = find_peaks(-self.df_5Y, width=5, distance=200, height=(x_std*(8-weight/100),x_std*8))
                self.low_peak_list_2 = peaks1
                self.low_peak_list_2 = self.low_peak_list_2.tolist()

                for i in range(0,len(self.low_peak_list_2)) :
                    if len(self.low_peak_list_2) > 6:
                        while self.low_peak_list_2[0] < self.just_TEI :
                            del self.low_peak_list_2[0]
                    for j in range(0, len(self.low_peak_list_2)) :
                        if len(self.low_peak_list_2) > 6:
                            while self.low_peak_list_2[-1] > self.just_TEO :
                                del self.low_peak_list_2[-1]
                    if len(self.low_peak_list_2) == int(self.TSP_count) :
                        break
                if len(self.low_peak_list_2) == int(self.TSP_count):
                    self.error_code_labeling = 1
                    break

            self.TSP_Center_list_2 = np.round((np.asarray(self.high_peak_list_2) + np.asarray(self.low_peak_list_2))/2).tolist()        

        except:
            for weight in range(0, 800, 1):
                peaks, properties = find_peaks(self.df_5Y, width=5, distance=200, height=(x_std*(8-weight/100)*2,x_std*8))
                self.high_peak_list_2 = peaks
                self.high_peak_list_2 = self.high_peak_list_2.tolist()

                for i in range(0,len(self.high_peak_list_2)) :
                    if len(self.high_peak_list_2) > 6:
                        while self.high_peak_list_2[0] < self.just_TEI :
                            del self.high_peak_list_2[0] 
                    for j in range(0, len(self.high_peak_list_2)) :
                        if len(self.high_peak_list_2) > 6:
                            while self.high_peak_list_2[-1] > self.just_TEO :
                                del self.high_peak_list_2[-1]
                    if len(self.high_peak_list_2) == int(self.TSP_count) :
                        break
                if len(self.high_peak_list_2) == int(self.TSP_count):
                    break

            for weight in range(0, 800, 1):
                peaks1, properties = find_peaks(-self.df_5Y, width=5, distance=200, height=(x_std*(8-weight/100)*2,x_std*8))
                self.low_peak_list_2 = peaks1
                self.low_peak_list_2 = self.low_peak_list_2.tolist()

                for i in range(0,len(self.low_peak_list_2)) :
                    if len(self.low_peak_list_2) > 6:
                        while self.low_peak_list_2[0] < self.just_TEI :
                            del self.low_peak_list_2[0]
                    for j in range(0, len(self.low_peak_list_2)) :
                        if len(self.low_peak_list_2) > 6:
                            while self.low_peak_list_2[-1] > self.just_TEO :
                                del self.low_peak_list_2[-1]
                    if len(self.low_peak_list_2) == int(self.TSP_count) :
                        break
                if len(self.low_peak_list_2) == int(self.TSP_count):
                    self.error_code_labeling = 1
                    break

            self.TSP_Center_list_2 = np.round((np.asarray(self.high_peak_list_2) + np.asarray(self.low_peak_list_2))/2).tolist()             
        print('   - find_TSP2 성공 (TSP_Center_list_2 완성)')
           
            
            
            
    ### target == 0 일때 사용됨 ###     
    # 1차 & 2차 결함영역 찾기    
    def find_defect(self):
        print('\033[1m' + '\033[95m' + '7. find_defect' + '\033[0m')
        first_defect_start = 0
        first_defect_end = 0
        second_defect_start = 0
        second_defect_end = 0
        second_defect_center = 0
        TEI_end = 0
        TEI_start = 0
        TEO_end = 0
        TEO_start = 0
        TSP_num = self.TSP_num.values
        Evaluation = self.Evaluation.values

        global TSP_end 
        global TSP_start
        
        # 결함 그룹별 결함 검출법 분류
        self.group1 = ['DFI', 'DFS', 'IDI', 'IDS' , 'DEP', 'DNG', 'PVN']   ##free span 영역 결함
        self.group2 = ['DNT', 'DSI', 'DSS']                                ##TSP 내부 영역 결함

       ####### Evaluation group 1 #######      
        if Evaluation in self.group1 :
            print('\033[1m' + '\033[93m' + '7.1  Evaluation group 1' + '\033[0m')
        
            if self.TSP_num.values == 'TEI' :
                print(f'   -  TSP_num : {TSP_num}, Eval : {self.Evaluation.values}입니다.')
#                 TEI_LEN = np.round((self.TEI_low - self.TEI_high)*3)

                ### TSP1번의 high점과 low점을 빼서서 2로나눈 거리에 1.5배한 거리 = LEN
                LEN = np.round((np.asarray(self.low_peak_list) - np.asarray(self.high_peak_list))/2*1.5) 
                first_defect_start = self.just_TEI
#                 first_defect_start = self.TEI_low + TEI_LEN + margin - 기존 코드

                ### TSP1번의 high점과 low점을 빼서서 2로나눈 거리에 1.5배한 거리를 TSP1의 high점에서 뺌
                first_defect_end = self.high_peak_list[0] - LEN[0]
                
            ###현재 TEO임에도 TEI와 동일하게 코드 짜여있네
            ###추후 정리 필요...
            elif self.TSP_num.values == 'TEO' :
                print(f'   -  TSP_num : {TSP_num}, Eval : {self.Evaluation.values}입니다.')
#                 TEI_LEN = np.round((self.TEI_low - self.TEI_high)*3)
                LEN = np.round((np.asarray(self.low_peak_list) - np.asarray(self.high_peak_list))/2*1.5)
    
                first_defect_start = self.just_TEI
                TSP_start = self.high_peak_list[0] - LEN[0]
                first_defect_end = TSP_start
                              
            # TEO라고 적혀있지만 실제로 TEI인 경우가 있음
            # elif self.TSP_num.values == 'TEO' : 
            #     TEO_LEN = (self.TEO_low - self.TEO_high)*1.5
            #     LEN = np.round((np.asarray(self.low_peak_list) - np.asarray(self.high_peak_list))/2*1.5)                
            #     first_defect_start = self.low_peak_list[-1] + LEN[-1]
            #     if TEO_LEN < 150 :
            #         first_defect_end = self.TEO_high - TEO_LEN - 150

            # TSP num = [1 : TEO]인 경우
            
            else :
                ###LEN = low와 high점을 빼서 2로나눈 거리에 2배한 거리
                print(f'   - TSP_num : {TSP_num}, Eval : {self.Evaluation.values}입니다.')
                LEN = np.round((np.asarray(self.low_peak_list) - np.asarray(self.high_peak_list))/2*2)
                
                ##T#SP_num 번째 TSP의 시작점(TSP_start)과 끝점(TSP_end), high peak가 low peak보다 앞쪽
                TSP_start = self.high_peak_list[int(TSP_num)-1] - LEN[int(TSP_num)-1]
                TSP_end = self.low_peak_list[int(TSP_num)-1] + LEN[int(TSP_num)-1]
                
                # 추가수정 필요 (TEO의 신호특성 분석 필요, 타 채널로 TEO 검증필요)
                # TEO_start = self.TEO_high - np.round((self.TEO_high - self.TEO_low)*2)  - 기존코드
                
                TEO_start = self.just_TEO
                print(f'   - TEO_start :  {TEO_start}')
                
                ###마지막 TSP일 때, 1차 영역은 마지막 TSP부터 TEO까지
                if TSP_num == self.TSP_count : 
                    first_defect_start = TSP_end
                    first_defect_end = TEO_start
                    
                ###마지막 TSP가 아닐 때,     
                else :
                    first_defect_start = TSP_end
                    first_defect_end = np.asarray(self.high_peak_list[int(TSP_num)]) - LEN[int(TSP_num)]
            self.TSP_start_graph = self.high_peak_list - LEN
            self.TSP_end_graph = self.low_peak_list + LEN
            print(f'   - [최종 1차 결함영역] : {first_defect_start} ~ {first_defect_end}')
       ####### Evaluation group 1의 1차 영역 완성 ####### 

    
       ####### Evaluation group 1의 2차 영역 찾기 시작 #######  
            ### CH1Y, CH3Y 두 채널을 더하고(diff_CH13Y_abs) 최댓값을 갖는 index 찾기
            diff_CH1Y_abs = final_data['CH1Y'][int(first_defect_start):int(first_defect_end)].diff().abs()
            diff_CH3Y_abs = final_data['CH3Y'][int(first_defect_start):int(first_defect_end)].diff().abs()

            diff_CH13Y_abs = np.array(final_data['CH1Y'].diff().abs()) + np.array(final_data['CH3Y'].diff().abs())
            diff_CH13Y_abs = pd.DataFrame(diff_CH13Y_abs)

            diff_CH13Y_abs_idxmax = diff_CH13Y_abs[int(first_defect_start):int(first_defect_end)].idxmax()
            diff_CH13Y_abs_idxmax.tolist()
            self.diff_CH13Y_idxmax = diff_CH13Y_abs_idxmax[0]

            ### 추후 cal_degree 에서 사용            
            self.diff_CH1Y_max = diff_CH1Y_abs[diff_CH1Y_abs == diff_CH1Y_abs.max()].index[0]
            self.diff_CH3Y_max = diff_CH3Y_abs[diff_CH3Y_abs == diff_CH3Y_abs.max()].index[0]
            self.diff_CH13Y_max = diff_CH13Y_abs[int(first_defect_start):int(first_defect_end)].max()     
            
            
            ### CH1Y의 최댓값이 CH3Y의 최댓값보다 크면 CH1Y+3Y기준 +- 25 구간 내 최댓값, 최솟값을 2차영역 high, low로 설정
            ### CH1Y, CH3Y 중 값이 큰(dominant) 채널을 기준으로 인덱스 찾기
            if diff_CH1Y_abs.max() > diff_CH3Y_abs.max() :
                second_defect_high = final_data['CH1Y'][self.diff_CH13Y_idxmax-25:self.diff_CH13Y_idxmax+25].idxmax()
                second_defect_low = final_data['CH1Y'][self.diff_CH13Y_idxmax-25:self.diff_CH13Y_idxmax+25].idxmin()

            if diff_CH3Y_abs.max() > diff_CH1Y_abs.max() :                                
                second_defect_high = final_data['CH3Y'][self.diff_CH13Y_idxmax-25:self.diff_CH13Y_idxmax+25].idxmax()
                second_defect_low = final_data['CH3Y'][self.diff_CH13Y_idxmax-25:self.diff_CH13Y_idxmax+25].idxmin()
                self.find_cal_degree = "CH3"
                self.error_code.append(['CH3번으로 라벨링'])

            ### 2차 결함영역에서 high가 low보다 클때,
            self.second_defect_length = int(abs(second_defect_high - second_defect_low)/2*1.5)             
            self.second_defect_end = second_defect_high + self.second_defect_length
            self.second_defect_start = second_defect_low - self.second_defect_length

            ### 2차 결함영역에서 low가 high보다 클때,
            if second_defect_high < second_defect_low :
                self.second_defect_end = second_defect_low + self.second_defect_length
                self.second_defect_start = second_defect_high - self.second_defect_length
       ####### Evaluation group 1, 2차 영역 완성 #######  
            print(f'   - Eval group1의 2차 결함영역 : {self.second_defect_start} ~ {self.second_defect_end}')

        
       ####### Evaluation group 2 #######    
        ### 대전제 : low_peak 가 high_peak보다 뒤쪽이다
        elif Evaluation in self.group2 :
            print('\033[1m' + '\033[93m' + '7.2  Evaluation group 2' + '\033[0m')
            ### df_CH7Y로 만들어진 peak_list, LEN_1 = (low-high)/2, lenght_1 = low-high 길이의 2배
            LEN_1 = np.round((np.asarray(self.low_peak_list[int(TSP_num)-1]) - np.asarray(self.high_peak_list[int(TSP_num)-1]))/2) 
            length_1 = abs(int(self.low_peak_list[int(TSP_num)-1] + LEN_1) - int(self.high_peak_list[int(TSP_num)-1] - LEN_1))
                        
            ### df_CH5Y로 만들어진 peak_list_2, LEN_2 = (low-high)/2, lenght_2 = low-high 길이의 2배
            LEN_2 = np.round((np.asarray(self.low_peak_list_2[int(TSP_num)-1]) - np.asarray(self.high_peak_list_2[int(TSP_num)-1]))/2)
            length_2 = abs(int(self.low_peak_list_2[int(TSP_num)-1] + LEN_2) - int(self.high_peak_list_2[int(TSP_num)-1] - LEN_2))

            if length_1 >= length_2 :
                self.second_defect_start = int(self.high_peak_list[int(TSP_num)-1] - LEN_1)
                self.second_defect_end = int(self.low_peak_list[int(TSP_num)-1] + LEN_1)                
            else :
                self.second_defect_start = int(self.high_peak_list_2[int(TSP_num)-1] - LEN_2)
                self.second_defect_end = int(self.low_peak_list_2[int(TSP_num)-1] + LEN_2)
                print('   - Error_Code = TSP 인식 부적절하여 5번채널로 인식', length_1, length_2)
                self.error_code.append('   - TSP인식 CH5Y 채널로 변경')
       ####### Evaluation group 2의 2차 영역 완성 #######   
            print(f'   - Eval group2의 2차 결함영역완성 : {self.second_defect_start} ~ {self.second_defect_end}')
        
#             diff_CH1Y_abs = final_data['CH1Y'][int(second_defect_start):int(second_defect_end)].diff().abs()
#             diff_CH3Y_abs = final_data['CH3Y'][int(second_defect_start):int(second_defect_end)].diff().abs()
#             self.diff_CH1Y_max = diff_CH1Y_abs[diff_CH1Y_abs == diff_CH1Y_abs.max()].index[0]
#             self.diff_CH3Y_max = diff_CH3Y_abs[diff_CH3Y_abs == diff_CH3Y_abs.max()].index[0]

       ####### Evaluation group 1,2 에 해당하지 않을 때 #######    
        else :
            print('Eval group 1, 2에 해당하지 않는 결함입니다.')
            self.error_code.append('Eval group 1, 2에 해당하지 않는 결함입니다.')
       ####### Evaluation group 1,2 에 해당하지 않음 #######   
       
       ####### 결합그룹 1, 2 찾기를 통해 만들어진 2차 결함영역 재정리 #######
      # low가 high보다 크다는 대전제가 틀렸다면 start와 end의 위치 바꾸기 (계산해보면 완벽)
        if self.second_defect_start > self.second_defect_end :
            temp_end = self.second_defect_start 
            self.second_defect_start = self.second_defect_end
            self.second_defect_end = temp_end

      # 2차결함영역이 너무 짧은 경우 보강
        self.second_defect_length = self.second_defect_end - self.second_defect_start
        if self.second_defect_length < 25 :
            self.second_defect_start = self.second_defect_start - 4
            self.second_defect_end = self.second_defect_end + 4
            self.second_defect_length = self.second_defect_end - self.second_defect_start
            self.error_code.append('2차영역이 25보다 작아 4씩 좌우로 보강')
        print(f'   - 2차 결함영역 : {self.second_defect_start} ~ {self.second_defect_end}, 2차결함길이 : {self.second_defect_length}')
    ### def find_defect (1차 & 2차 결함영역 찾기) 완성 ###    

    
    
        ####### Evaluation group 1, 2의 2차 영역 완성본의 길이 조절하기 ########
        self.second_defect_center = self.second_defect_start + np.round((self.second_defect_end - self.second_defect_start)/2)
        self.second_defect_start = int(self.second_defect_center - self.required_length/2)
        self.second_defect_end = int(self.second_defect_center + self.required_length/2)
        self.second_defect_length = int(self.second_defect_end - self.second_defect_start)
        
        print(f'   - [최종 2차 결함영역] : {self.second_defect_start} ~ {self.second_defect_end}')
        print(f'   - [최종 2차 결함영역 길이] : {self.second_defect_length}')
    
    
        
    ### target == 2 일때 사용됨 ###
    ### 정상 tsp 구간 찾고자 함 ###
    def get_tsp(self) :
        print('\033[1m' + '\033[95m' + '7. get_tsp (정상 tsp 영역 찾기)' + '\033[0m')

        TSP_num = random.randint(2, 20)
        
        if TSP_num == self.TSP_num.values:
            TSP_num = self.TSP_num.values - 1

        if TSP_num > self.TSP_count - 2 :   #랜덤으로 고른 TSP_num이 TEO쪽 끝 2개라면, 
            TSP_num = TSP_num - 4   # TSP_num에 -4한다.
        print('   - TSP ', TSP_num, '번 데이터를 취득합니다.')        
        
        ### Evaluation in self.group2에서 사용된 코드와 동일 LEN_1, length_1, LEN_2, length_2
        LEN_1 = np.round((np.asarray(self.low_peak_list[int(TSP_num)-1]) - np.asarray(self.high_peak_list[int(TSP_num)-1]))/2)       
        length_1 = abs(int(self.low_peak_list[int(TSP_num)-1] + LEN_1) - int(self.high_peak_list[int(TSP_num)-1] - LEN_1))

        LEN_2 = np.round((np.asarray(self.low_peak_list_2[int(TSP_num)-1]) - np.asarray(self.high_peak_list_2[int(TSP_num)-1]))/2)       
        length_2 = abs(int(self.low_peak_list_2[int(TSP_num)-1] + LEN_2) - int(self.high_peak_list_2[int(TSP_num)-1] - LEN_2))

        if length_1 >= length_2 :
            self.second_defect_start = int(self.high_peak_list[int(TSP_num)-1] - LEN_1)
            self.second_defect_end = int(self.low_peak_list[int(TSP_num)-1] + LEN_1)                
        else :
            self.second_defect_start = int(self.high_peak_list_2[int(TSP_num)-1] - LEN_2)
            self.second_defect_end = int(self.low_peak_list_2[int(TSP_num)-1] + LEN_2)
            print('   - Error_Code = TSP 인식 부적절하여 5번채널로 인식', length_1, length_2)
            self.error_code.append('   - TSP인식 CH5Y 채널로 변경')
        
        
        ####### 정상 tsp의 영역 길이 조절하기 ########
        self.second_defect_center = self.second_defect_start + np.round((self.second_defect_end - self.second_defect_start)/2)
        self.second_defect_start = int(self.second_defect_center - self.required_length/2)
        self.second_defect_end = int(self.second_defect_center + self.required_length/2)
        self.second_defect_length = int(self.second_defect_end - self.second_defect_start)
        
        self.tsp_result = np.array(final_data[self.second_defect_start : self.second_defect_end])   #실제 tsp 결과 데이터
        self.length = self.second_defect_end - self.second_defect_start   #실제 tsp 구간 length       
        
        print(f'   - [최종 정상 TSP] : {self.second_defect_start} ~ {self.second_defect_end}')
        print(f'   - [최종 정상 TSP 길이] : {self.second_defect_length}')
        
        
        
        
    ### target == 1 일때 사용됨 ###
    ### 정상영역 전열관 찾기 ###
    def get_normal(self, max_len= 120) :
        print('\033[1m' + '\033[95m' + '7. get_normal (정상 Freespan 영역 찾기)' + '\033[0m')
        TSP_num = random.randint(2, 20)     
        if TSP_num == int(self.TSP_num.values):  #랜덤 TSP가 결함위치가 같다면,
            TSP_num = int(self.TSP_num.values) -2
            if TSP_num < 1:
                TSP_num = TSP_num+10
        print('   - 정상영역', TSP_num, '~', TSP_num+1, '번 구간의 데이터를 추출합니다.')
        
        LEN = np.round((np.asarray(self.low_peak_list) - np.asarray(self.high_peak_list))/2*2)
        
        ####### get_normal의 영역 길이 조절하기 ########
        self.second_defect_start = int(np.asarray(self.low_peak_list[int(TSP_num)]) + LEN[int(TSP_num)])+500
        self.second_defect_end = int(self.second_defect_start + self.required_length)
        
        self.normal_result =  np.array(final_data[self.second_defect_start : self.second_defect_end])
        self.length = self.second_defect_end - self.second_defect_start   #실제 정상영역 구간 length       

        print(f'   - [최종 정상 전열관] :  {self.second_defect_start} ~ {self.second_defect_end}')
        print(f'   - [최종 정상 전열관 길이] :  {self.length}')
        
        
    def show_fig(self, df):
        fig = px.line(df, markers=True, width=1200, height =800, title=filename)
        fig.show()

        
    #Strip(TEI, TEO, TSP) 그리기
    def get_TE(self, df):
        x_std = np.std(self.df_7Y)
        fig = px.line(df, markers=True, width=1000, height =800, title=filename)
        fig.add_scatter(x=df.index, y=df.CH1Y.diff(), mode='lines', name='CH1Y diff')
        fig.add_scatter(x=df.index, y=df.CH3Y.diff(), mode='lines', name='CH3Y diff')
        fig.add_scatter(x=df.index, y=df.CH5Y.diff(), mode='lines', name='CH5Y diff')
        fig.add_scatter(x=df.index, y=df.CH7Y.diff(), mode='lines', name='CH7Y diff')

        fig.add_vline(x=self.just_TEI, line_width=0.5, line_dash="solid", line_color="red", annotation_text="just_TEI", annotation_position="top")        
        fig.add_vline(x=self.just_TEO, line_width=0.5, line_dash="solid", line_color="red", annotation_text="just_TEO", annotation_position="top")  

#         fig.add_hline(y=x_std * 7, line_width=0.5, line_dash="solid", line_color="red")
#         fig.add_hline(y=-x_std * 7, line_width=0.5, line_dash="solid", line_color="red")
        fig.add_hrect(y0=-x_std * 7, y1= x_std * 7, line_width=0, fillcolor="red", opacity=0.1, annotation_text="x_std*7", annotation_position="top left")              
                      
        fig.add_hline(y=self.df_7Y.max()*0.8, line_width=1, line_dash="solid", line_color="blue", name='df_7Y.max()*0.8')
        
#         fig.add_vline(x=self.second_defect_start, line_width=1, line_dash="solid", line_color="blue", name='second_defect_start')  
#         fig.add_vline(x=self.second_defect_end, line_width=1, line_dash="solid", line_color="blue", name='second_defect_end') 
        fig.add_vrect(x0=self.second_defect_start, x1= self.second_defect_end, line_width=0.5, fillcolor="green", opacity=0.5, annotation_text="second_defect 구간", annotation_position="top")         
        
        i=1              
        for TSP in self.TSP_Center_list :
            fig.add_vline(x=TSP, line_width=0.5, line_dash="solid", line_color="purple",
                          annotation_text=f"TSP_{i}_center",
                          annotation_position="right",
                          annotation_font_size=5,
                          annotation_font_color="blue")
            i= i+1
        fig.show()

        
    def distance_max(self, channel='CH7', size=700) :
        target_df = final_data[self.second_defect_start:self.second_defect_end]
        distance_max = 0
        distance_max_list = []
        target_df_x = final_data[channel+'X'][self.second_defect_start:self.second_defect_end]
        target_df_y = final_data[channel+'Y'][self.second_defect_start:self.second_defect_end]
        
        if self.second_defect_length < 200 :
        
            for i in range (0,len(target_df_x)):
                x1 = final_data[channel+'X'][self.second_defect_start:self.second_defect_end][self.second_defect_start+i]
                y1 = final_data[channel+'Y'][self.second_defect_start:self.second_defect_end][self.second_defect_start+i]

                for j in range (0,len(target_df_x)):
                    x2 = final_data[channel+'X'][self.second_defect_start:self.second_defect_end][self.second_defect_start+j]
                    y2 = final_data[channel+'Y'][self.second_defect_start:self.second_defect_end][self.second_defect_start+j]

                    distance = ( (x2 - x1) **2 + (y2 - y1) **2 )**0.5

                    if distance > distance_max :
                        distance_max = distance
                        distance_max_list.append([distance, self.second_defect_start+i, self.second_defect_start+j])

            distance_max_list = pd.DataFrame(distance_max_list)      
            distance_max_list.columns = ['distance', 'max_point', 'min_point']
            max_point = distance_max_list.loc[(len(distance_max_list)-1), 'max_point']
            min_point = distance_max_list.loc[(len(distance_max_list)-1), 'min_point']
            x1 = final_data[channel+'X'][self.second_defect_start:self.second_defect_end][max_point]
            y1 = final_data[channel+'Y'][self.second_defect_start:self.second_defect_end][max_point]
            x2 = final_data[channel+'X'][self.second_defect_start:self.second_defect_end][min_point]
            y2 = final_data[channel+'Y'][self.second_defect_start:self.second_defect_end][min_point]
            # print('x1,x2,y1,y2', x1, y1, x2, y2)
            # print('max_point', max_point)
            # print('min_point', min_point)
            # print('Peak to Peak Point', max_point, min_point)

            self.con_x = [x1, x2]
            self.con_y = [y1, y2]

            degree = math.atan((y2-y1)/(x1-x2))
            self.peak_degree = math.degrees(degree)
            self.max_length =  ((x2- x1)**2 + (y2-y1)**2 )**0.5
            print('   - [max dgree] : ', self.peak_degree)
            
        else :
            print('   - 2차결함영역이 잘못 설정되었습니다.')
            self.max_length = 9999
            self.error_code.append('max_length error')
            

    def cal_degree(self) :
        if self.Evaluation.values in self.group2 :
            self.dgree_CH1 = 0
            self.dgree_CH3 = 0

        elif self.Evaluation.values in self.group1 :
            max1 = self.diff_CH1Y_max
            max3 = self.diff_CH3Y_max
            
            if self.find_cal_degree == "CH3" :
                max1 = max3

            CH1_x1 = final_data['CH1X'][max1-1]
            CH1_y1 = final_data['CH1Y'][max1-1]
            CH1_x2 = final_data['CH1X'][max1]
            CH1_y2 = final_data['CH1Y'][max1]

            CH3_x1 = final_data['CH3X'][max1-1]
            CH3_y1 = final_data['CH3Y'][max1-1]
            CH3_x2 = final_data['CH3X'][max1]
            CH3_y2 = final_data['CH3Y'][max1]

            self.max_point_x_CH1 = [CH1_x1, CH1_x2]
            self.max_point_y_CH1 = [CH1_y1, CH1_y2]
            self.max_point_x_CH3 = [CH3_x1, CH3_x2]
            self.max_point_y_CH3 = [CH3_y1, CH3_y2]

            self.dgree_CH1 = math.degrees(math.atan((CH1_y2-CH1_y1)/(CH1_x1-CH1_x2)))
            self.dgree_CH3 = math.degrees(math.atan((CH3_y2-CH3_y1)/(CH3_x1-CH3_x2)))

            if self.Evaluation.values == 'DNG' :
                self.dgree_CH1 = 180+math.degrees(math.atan((CH1_y1-CH1_y2)/(CH1_x2-CH1_x1)))

        print('   - CH1(계산값):', self.dgree_CH1, '\n''     CH3(계산값):', self.dgree_CH3, '\n''     실제값 :' ,self.Dgree.values)

        
    def get_lissajous(self, channel='CH7'):
        target_df = final_data[self.second_defect_start:self.second_defect_end]
#         if self.Evaluation.values in self.group1 :
#             if channel == 'CH1' :
#                 self.max_point_x = self.max_point_x_CH1
#                 self.max_point_y = self.max_point_y_CH1
#             elif channel == 'CH3' :
#                 self.max_point_x = self.max_point_x_CH3
#                 self.max_point_y = self.max_point_y_CH3
#         elif self.Evaluation.values in self.group2 :
#                 self.max_point_x = self.con_x
#                 self.max_point_y = self.con_y
        print(f'target_df =', {channel})
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                name="lissajous_chart",
                x=target_df[channel+'X'],
                y=target_df[channel+'Y'],
                hovertext=target_df.index,
                mode='markers+lines',
                marker=dict(size=5, line_width=0.5)
            )
        )
        
#         fig.add_trace(
#             go.Scattergl(
#                 name="Peak to Peak",
#                 x=self.max_point_x,
#                 y=self.max_point_y,
#                 hovertext=target_df.index,
#                 mode='markers',
#                 marker=dict(size=5, line_width=0.1)
#             )
#         )
        
#  MAX MIN  점 표시
        fig.add_trace(
            go.Scattergl(
                name="Peak to Peak",
                x=self.con_x,
                y=self.con_y,
                hovertext=target_df.index,
                mode='markers',
                marker=dict(size=5, line_width=0.1, color='red')
            )
        )
    
#         fig.add_trace(
#             go.Scattergl(
#                 name="ymax or ymin",
#                 x = con_x,
#                 y = con_y,
#                 mode='markers',
#                 marker=dict(size=3, line_width=0, color='red')
#             )
#         )

        fig.update_layout(
            width=700, height=700
        )
    
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1
        )
        
        fig.show()
        print(os.path.join(self.output_path, f'{filename}_{channel}_{self.Evaluation.values}.png'))
#         print(f'{self.output_path}{filename}_{channel}_{self.Evaluation.values}.jpg')
#         fig.write_image(os.path.join(self.output_path, f'{filename}_{channel}_{self.Evaluation.values}.png'))
#         fig.write_html(f'{self.output_path}{filename}_{channel}_{self.Evaluation.values}.html')
#         pio.write_image(fig, f'{self.output_path}{filename}_{channel}_{self.Evaluation.values}.jpg')
        

    def show_plt(self) :
        print(target_df_x,target_df_y)
        plt.plot[target_df_x,target_df_y]
        plt.show()


    def dataset_second_defect(self):
        target = final_data
        target = np.array(target[self.second_defect_start : self.second_defect_end])
        self.result_data = target
### 라벨링 툴 Class 완료 ###

##########################################################################################

### 라벨링 툴 실행 함수 ###
def labeling(final_data, master_df, filename, path, target):  
    labeling = Labeling_Tool(path)
#     labeling.show_fig(final_data)
    labeling.find_TEITEO_Candidate()     #TEI, TEO 후보 peaks 찾기 (CH7Y)
    labeling.find_TEI_TEO_high_low()     #후보 중 가장 긴 구간(실제 DATA 존재구간) 찾기
    labeling.find_just_TEI()             #TEI를 쩌스트 하게 짜르기 (CH1Y)
    labeling.find_just_TEO()             #TEO를 쩌스트하게 짜르기 (CH1Y)
    labeling.find_TSP()                  #TSP 양끝 및 중심 찾기 (CH7Y)
    labeling.find_TSP_2()                #TSP Center 찾기 (CH5Y)

    print('\033[92m' + '\033[1m' +  '    target = '+ '\033[0m', target)       

    if target == 0:   #결함찾기
        labeling.find_defect()   #출력이 self.second_defect_start, self.second_defect_end

        labeling.distance_max('CH1')
        labeling.cal_degree()

        labeling.dataset_second_defect()  #출력이 self.result_data = final_data[self.second_defect_start : self.second_defect_end]
        labeling.Eval = labeling.Evaluation.values
        print('   - labeling.error_code : ',labeling.error_code)
#         print('\033[92m' + '\033[1m' +  '    target = '+ '\033[0m', target '완료')  
        
    elif target == 1:   #정상 전열관 데이터 가져오기
        labeling.get_normal(50)  #출력이 normal_result 
        labeling.second_defect_length = labeling.length   #length 는 2차 결함영역 (get_normal)
        labeling.result_data = labeling.normal_result   #normal_result에 margin을 더한 구간
        labeling.dgree_CH1= 0
        labeling.peak_degree = 0
        labeling.max_length = 0
        print(labeling.Dgree.values)
        labeling.Eval = 'NDD'
#         print('\033[92m' + '\033[1m' +  '    target = '+ '\033[0m', target '완료')  
        
    elif target == 2:   #정상 TSP 데이터 가져오기
        labeling.get_tsp()   #출력이 tsp_result = TSP 정상구간 final_data
        labeling.second_defect_length = labeling.length   #length 는 2차 결함영역 (get_tsp)
        labeling.result_data = labeling.tsp_result   #tsp_result 는 2차 결함영역의 final data (get_tsp)
        labeling.dgree_CH1= 0
        labeling.peak_degree = 0
        labeling.Dgree.values
        labeling.max_length = 0
        labeling.Eval = 'TSP' 

    final_result = filename, labeling.result_data, labeling.Eval, labeling.second_defect_length, labeling.dgree_CH1, labeling.peak_degree, labeling.Dgree.values, labeling.max_length
    return final_result

def auto(final_data, filename, output_path):
        labeling = Labeling_Tool(output_path)
#     labeling.show_fig(final_data)
        labeling.find_TEITEO_Candidate()
        labeling.find_TEI_TEO_high_low()
        labeling.find_just_TEI()
        labeling.find_just_TEO()
        labeling.find_TSP()
        labeling.find_defect()
        real_data = final_data[labeling.just_TEI : labeling.just_TEO]
        center_list = labeling.TSP_Center_list
        TEI = labeling.just_TEI

        return real_data, center_list, TEI, labeling.second_defect_start, labeling.second_defect_end, labeling.TSP_start_graph, labeling.TSP_end_graph
        
def parsing(input_path, filename):
    print(f"\033[31m \033[43m{filename}  파싱 시작\033[0m")
    dp = DatParser(input_path, filename)
    dp.set_file()
    dp.get_channel_names()
    dp.get_setting()
    dp.parse()
    print(f"\033[31m \033[43m{filename}  파싱 완료\033[0m")
    return dp.parsing_data

def labeling(final_data, master_df, filename, path, target):  
    labeling = Labeling_Tool(path, master_df)
    labeling.find_TEITEO_Candidate()     #TEI, TEO 후보 peaks 찾기 (CH7Y)
    labeling.find_TEI_TEO_high_low()     #후보 중 가장 긴 구간(실제 DATA 존재구간) 찾기
    labeling.find_just_TEI()             #TEI를 쩌스트 하게 짜르기 (CH1Y)
    labeling.find_just_TEO()             #TEO를 쩌스트하게 짜르기 (CH1Y)
    labeling.find_TSP()                  #TSP 양끝 및 중심 찾기 (CH7Y)
    labeling.find_TSP_2()                #TSP Center 찾기 (CH5Y)
    print('\033[92m' + '\033[1m' +  '    target = '+ '\033[0m', target)       

    if target == 0:   #결함찾기
        labeling.find_defect()   #출력이 self.second_defect_start, self.second_defect_end

        labeling.distance_max('CH1')
        labeling.cal_degree()

        labeling.dataset_second_defect()  #출력이 self.result_data = final_data[self.second_defect_start : self.second_defect_end]
        labeling.Eval = labeling.Evaluation.values
        print('   - labeling.error_code : ',labeling.error_code)
#         print('\033[92m' + '\033[1m' +  '    target = '+ '\033[0m', target '완료')  
        
    elif target == 1:   #정상 전열관 데이터 가져오기
        labeling.get_normal(50)  #출력이 normal_result 
        labeling.second_defect_length = labeling.length   #length 는 2차 결함영역 (get_normal)
        labeling.result_data = labeling.normal_result   #normal_result에 margin을 더한 구간
        labeling.dgree_CH1= 0
        labeling.peak_degree = 0
        labeling.max_length = 0
        print(labeling.Dgree.values)
        labeling.Eval = 'NDD'
        
    elif target == 2:   #정상 TSP 데이터 가져오기
        labeling.get_tsp()   #출력이 tsp_result = TSP 정상구간 final_data
        labeling.second_defect_length = labeling.length   #length 는 2차 결함영역 (get_tsp)
        labeling.result_data = labeling.tsp_result   #tsp_result 는 2차 결함영역의 final data (get_tsp)
        labeling.dgree_CH1= 0
        labeling.peak_degree = 0
        labeling.Dgree.values
        labeling.max_length = 0
        labeling.Eval = 'TSP' 

    final_result = filename, labeling.result_data, labeling.Eval, labeling.second_defect_length,\
    labeling.dgree_CH1, labeling.peak_degree, labeling.Dgree.values, labeling.max_length
    
    return final_result

master_df = pd.read_csv("Master_df.csv")

dirname = "/data/dat/dat_files"

f = open('tmp.txt', 'r')
filename = f.readline()

pathname = filename[0:11]
input_path = os.path.join(dirname,filename + '.dat')

output_path = os.path.join('./', filename)
# output_path = './labeled_files'
try:
    os.mkdir(output_path)
except:
    pass
final_data = parsing(input_path, filename)
real_data, TSP_Center_list, TEI, defect_start, defect_end, tsp_start_graph, tsp_end_graph  = auto(final_data, filename, output_path)

real_data.to_csv(os.path.join(output_path,"real_data.csv"))
final_data.to_csv(os.path.join(output_path,"final_data.csv"))

f = open(os.path.join(output_path,"defect_location.txt"), 'w')
f.write(str(defect_start)+'\t'+str(defect_end))
f.close()

# dirname = "./dat_files"

# filename = "HB-01-23-CD-A1-01-039018"

# pathname = filename[0:11]
# input_path = os.path.join(dirname,filename+'.dat')
# output_path = './labeled_files'


# final_data = parsing(input_path, filename)
# real_data, TSP_Center_list, TEI, defect_start, defect_end, tsp_start_graph, tsp_end_graph  = auto(final_data, filename, output_path)
