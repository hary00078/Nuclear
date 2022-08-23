from distutils.command.install_lib import PYTHON_SOURCE_EXTENSION
import os
import numpy as np
import pandas as pd
import plotly.express as px
import coredotseries as cds
from coredotseries.series import find_break_points, find_local_peaks, get_peak_wave_interval
from custom import *

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from matplotlib import pyplot, transforms
import plotly.offline as pyo
from scipy.signal import find_peaks
from IPython.display import Image
from plotly.subplots import make_subplots

cds.util.plotly_offline_mode(True)
cds.util.ignore_warning()

class Visualize:
    def __init__(self, pns, defect_tool, defect_ai):
        self.pns = pns
        self.defect_tool = defect_tool
        self.defect_ai = defect_ai
        self.df = None
        self.tei = None
        self.teo = None
        self.tsp = None
        self.defect_channel = None
        self.defect_group = None
        self.support_start = None
        self.support_end = None
    
    def extract_feature(self):
        master = pd.read_csv("./Master_df.csv")
        idx = master.index[master["PNS No"] == self.pns].values[0]
        srs = master.iloc[idx]
        name = srs['PNS No']
        parsed_path = os.path.join(name, "final_data.csv")
        channel = srs['Eval.']
        support_name = srs['TSP']

        # Parsed Data 불러오기
        df = pd.read_csv(parsed_path)

        # 컬럼명 변경 : 구버전 컬럼명(D1~4, A1~4) -> 신버전 컬럼명(CH1~CH8)
        # df = rename_old_columns(df)
        channel_cols = df.columns

        # TEI, TEO, TSP 구간 구하기 : CH7Y 활용 -> 구조물 center 좌표
        tei, teo, tsp = find_structure(df, column_name='CH7Y', tsp_where='center')
        df = df[channel_cols]

        # 해당 Support 구간 구하기 -> 1차 영역
        support_start, support_end = find_support(support_name, tei, teo, tsp, tsp_margin=75, tubeend_margin=150)

        # 결함 종류에 따른 검출 채널 및 결함 그룹 설정
        detection_channel, defect_group = set_standard(srs['Eval.'])

        # 결함 그룹 별 결함 구간 구하기 -> 2차 영역
        defect_start, defect_end = find_defect(df, detection_channel, defect_group, support_start, support_end, tsp_margin=75, defect_margin=50)

        # 추가 구간 확인
        if defect_group == 1 or defect_group == 2:
            target_df = df[defect_start:defect_end]
            peak_x, interval_x = get_peak_wave_interval(target_df, detection_channel+"Y")
        else:
            peak_x, interval_x = None, None
        
        self.df = df
        self.support_start = support_start
        self.support_end = support_end
        self.tei, self.teo, self.tsp = tei, teo, tsp
        self.defect_channel, self.defect_group = detection_channel, defect_group
        
    def select_proper_defect_ai(self):
        return_tensor = []
        return_tsp_range = []
        for i, candidate in enumerate(self.defect_ai):
            if candidate[0] < self.tei:
                continue
            if candidate[1] > self.teo:
                break
            
            defect_start = candidate[0]
            defect_end = candidate[1]
            
            
            tsp_list = [self.tei] + self.tsp + [self.teo]
            for j, tsp in enumerate(tsp_list):
                if defect_end < tsp:
                    break
            
            tsp_range = (tsp_list[j-1], tsp_list[j])
            
            if defect_start in range(tsp_range[0]-50, tsp_range[0]+50):
                continue
            elif defect_end in range(tsp_range[1]-50, tsp_range[1]+50):
                continue
            
            return_tensor.append((defect_start, defect_end))
            return_tsp_range.append(tsp_range)
            
        return return_tensor, return_tsp_range
            
    def visualize_all(self) :
        try:
            os.makedirs(os.path.join('./', self.pns, 'picture'))
        except:
            pass
        print("------------------------------")
        print("---------Making Picture-------")
        print("------------------------------")
        df = self.df
        fig = px.line(df, markers=True, width=1600, height=1000, title=self.pns)
        fig.add_scatter(x=df.index, y=df.CH1Y.diff(), mode='lines', name='CH1Y diff')
        fig.add_scatter(x=df.index, y=df.CH3Y.diff(), mode='lines', name='CH3Y diff')
        fig.add_scatter(x=df.index, y=df.CH5Y.diff(), mode='lines', name='CH5Y diff')
        fig.add_scatter(x=df.index, y=df.CH7Y.diff(), mode='lines', name='CH7Y diff')

        # TEI, TEO (빨강)
        fig.add_vline(x=self.tei, line_width=1, line_dash="solid", line_color="red")
        fig.add_vline(x=self.teo, line_width=1, line_dash="solid", line_color="red")

        # TSP (보라)
        for peak in self.tsp:
            fig.add_vline(x=peak, line_width=1, line_dash="solid", line_color="purple")

        if self.defect_group == 1:
            # 1차 영역 (파랑)
            fig.add_vline(x=self.support_start, line_width=1, line_dash="solid", line_color="blue")
            fig.add_vline(x=self.support_end, line_width=1, line_dash="solid", line_color="blue")

        # 2차 영역 (초록)
        defect_start = self.defect_tool[0]
        defect_end = self.defect_tool[1]
        fig.add_vline(x=defect_start, line_width=1, line_dash="solid", line_color="green")
        fig.add_vline(x=defect_end, line_width=1, line_dash="solid", line_color="green")
        
        fig.write_image(os.path.join('./', self.pns, 'picture', 'all.png'))
        
        return
    
    def visulalize_defect(self):
        try:
            os.makedirs(os.path.join('./', self.pns, 'picture','tool'))
        except:
            pass
        try:
            os.makedirs(os.path.join('./', self.pns, 'picture','ai'))
        except:
            pass
        
        defect_start = self.defect_tool[0]
        defect_end = self.defect_tool[1]
        fig = get_lissajous(self.df, defect_start, defect_end, channel=self.defect_channel, verbose=True)
        fig.write_image(os.path.join('./', self.pns, 'picture','tool', 'defect_tool.png'))
        
        for idx, candidate in enumerate(self.defect_ai):
            try:
                os.makedirs(os.path.join('./', self.pns, 'picture','ai', str(idx+1)))
            except:
                pass
            defect_start = candidate[0]
            defect_end = candidate[1]
            for i in range(1,9):
                fig = get_lissajous(self.df, defect_start, defect_end, channel='CH'+str(i), verbose=True)
                fig.write_image(os.path.join('./', self.pns, 'picture','ai', str(idx+1), f'channel={i}.png'))
        return
    
    def visulalize_report(self):
        try:
            os.makedirs(os.path.join('./', self.pns, 'picture','tool'))
        except:
            pass
        try:
            os.makedirs(os.path.join('./', self.pns, 'picture','ai'))
        except:
            pass
        defect_start = self.defect_tool[0]
        defect_end = self.defect_tool[1]
        fig = get_lissajous(self.df, defect_start, defect_end, channel=self.defect_channel, verbose=True)
        fig.write_image(os.path.join('./', self.pns, 'picture','tool', 'defect_tool.png'))
        
        return_tensor, return_tsp_range = self.select_proper_defect_ai()
        iter_num = len(return_tensor)
        numbering = 1
        for idx in range(iter_num):
            defect_candi = return_tensor[idx]
            tsp_range = return_tsp_range[idx]
            
            defect_start = defect_candi[0]
            defect_end = defect_candi[1]
            
            target_df = self.df[defect_candi[0]:defect_candi[1]]
            
            degree_1 = math.degrees(math.atan((max(target_df['CH1Y'])-min(target_df['CH1Y']))/(max(target_df['CH1X'])-min(target_df['CH1X']))))
            degree_3 = math.degrees(math.atan((max(target_df['CH3Y'])-min(target_df['CH3Y']))/(max(target_df['CH3X'])-min(target_df['CH3X']))))
            degree_5 = math.degrees(math.atan((max(target_df['CH5Y'])-min(target_df['CH5Y']))/(max(target_df['CH5X'])-min(target_df['CH5X']))))
            degree_7 = math.degrees(math.atan((max(target_df['CH7Y'])-min(target_df['CH7Y']))/(max(target_df['CH7X'])-min(target_df['CH7X']))))
            
            try:
                os.makedirs(os.path.join('./', self.pns, 'picture','ai', str(numbering)))
            except:
                pass                
            
            fig = make_subplots(rows=3, cols=2, specs=[[{}, {}],[{}, {}],\
                                [{"colspan":2},None]],subplot_titles=(str(degree_1), str(degree_3), str(degree_5), str(degree_7), "2nd defect location of CH1Y"))
            fig.add_trace(
                go.Scattergl(
                    name="lissajous_chart1",
                    x=target_df['CH'+str(1)+'X'],
                    y=target_df['CH'+str(1)+'Y'],
                    hovertext=target_df.index,
                    mode='markers+lines',
                    marker=dict(size=5, line_width=0.5)
                ),
                row=1, col=1)
            fig.add_trace(
                go.Scattergl(
                    name="lissajous_chart3",
                    x=target_df['CH'+str(3)+'X'],
                    y=target_df['CH'+str(3)+'Y'],
                    hovertext=target_df.index,
                    mode='markers+lines',
                    marker=dict(size=5, line_width=0.5)
                ),
                row=1, col=2)
            fig.add_trace(
                go.Scattergl(
                    name="lissajous_chart5",
                    x=target_df['CH'+str(5)+'X'],
                    y=target_df['CH'+str(5)+'Y'],
                    hovertext=target_df.index,
                    mode='markers+lines',
                    marker=dict(size=5, line_width=0.5)
                ),
                row=2, col=1)
            fig.add_trace(
                go.Scattergl(
                    name="lissajous_chart7",
                    x=target_df['CH'+str(7)+'X'],
                    y=target_df['CH'+str(7)+'Y'],
                    hovertext=target_df.index,
                    mode='markers+lines',
                    marker=dict(size=5, line_width=0.5)
                ),
                row=2, col=2)
            fig.add_trace(
                go.Scattergl(
                    name="2nd defect location of CH1Y",
                    x=self.df[tsp_range[0]:tsp_range[1]].index.tolist(),
                    y=self.df[tsp_range[0]:tsp_range[1]]["CH1"+'Y'],
                    hovertext=self.df[defect_start:defect_end].index,
                    mode='markers+lines',
                    marker=dict(size=2, line_width=0.5)
                ), 
                row=3, col=1)
            fig.add_vrect(x0=defect_start, x1=defect_end, fillcolor='red', opacity=0.5, row=3, col=1, line_width=0)
            fig.add_vline(x=tsp_range[0], line_width=0.8, line_dash="solid", line_color="purple",
                        annotation_text=f"{tsp_range[0]}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color="blue", row=3, col=1)
            fig.add_vline(x=tsp_range[1], line_width=0.8, line_dash="solid", line_color="purple",
                        annotation_text=f"{tsp_range[1]}",
                        annotation_position="right",
                        annotation_font_size=10,
                        annotation_font_color="blue", row=3, col=1)
            fig.update_layout(title=f'TSP - TSP : {tsp_range[0]} - {tsp_range[1]} || Predicted defect location : {defect_start} - {defect_end}')
            fig.write_image(os.path.join('./', self.pns, 'picture','ai', str(numbering), f'report.png'))
            numbering += 1
        return return_tensor