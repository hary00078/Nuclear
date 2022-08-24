import itertools
import math
import operator
import re
from turtle import title
import plotly.graph_objects as go
from coredotseries.series import find_local_peaks, binning, find_longest_element


def find_structure(df, column_name='D4Y', order=800, bins=4, tsp_where='low', tsp_length_opt=True):
    # find local peaks
    peak_low, peak_high = find_local_peaks(df, column_name=column_name, order=order, verbose=False, number=500, idx_start=None, idx_end=None, std_weight=0.9)

    # 인접한 여러 인덱스가 Peak로 나타날 수 있어서 인덱스가 직전과 동일하면 직전 인덱스 삭제
    # 인접한 인덱스 중 마지막 인덱스만 남김
    for idx in peak_low.index:
        if idx-1 in peak_low.index:
            peak_low = peak_low.drop(idx-1)
    for idx in peak_high.index:
        if idx-1 in peak_high.index:
            peak_high = peak_high.drop(idx-1)

    # bin peak_low and peak_high
    peak_low_bin = binning(peak_low[column_name], bins=bins)
    peak_high_bin = binning(peak_high[column_name], bins=bins)

    # split list into indices based on consecutive identical value
    peak_low_split = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(peak_low_bin), key=operator.itemgetter(1))]
    peak_high_split = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(peak_high_bin), key=operator.itemgetter(1))]

    # find TSP index (longest indices)
    tsp_low_idx = find_longest_element(peak_low_split)
    tsp_high_idx = find_longest_element(peak_high_split)

    # find peak (TEI candidate)
    NUMBER_SMALL = 0
    try:
        tei_low = peak_low.index[tsp_low_idx[0] - 1]
    except IndexError:
        tei_low = NUMBER_SMALL
    try:
        tei_high = peak_high.index[tsp_high_idx[0] - 1]
    except IndexError:
        tei_high = NUMBER_SMALL

    # find peak (TEO candidate)
    NUMBER_BIG = 10e10
    try:
        teo_low = peak_low.index[tsp_low_idx[-1] + 1]
    except IndexError:
        teo_low = NUMBER_BIG
    try:
        teo_high = peak_high.index[tsp_high_idx[-1] + 1]
    except IndexError:
        teo_high = NUMBER_BIG

    # find peak (TSP candidate)
    tsp_low = [peak_low.index[idx] for idx in tsp_low_idx]
    tsp_high = [peak_high.index[idx] for idx in tsp_high_idx]

    # option : make same length between tsp_low and tsp_high
    # only change TEO side (fixed TEI)
    if tsp_length_opt:
        if len(tsp_low) > len(tsp_high):
            teo_low = tsp_low[min(len(tsp_low), len(tsp_high))]
            tsp_low = tsp_low[:min(len(tsp_low), len(tsp_high))]
        elif len(tsp_low) < len(tsp_high):
            teo_high = tsp_high[min(len(tsp_low), len(tsp_high))]
            tsp_high = tsp_low[:min(len(tsp_low), len(tsp_high))]

    # set TEI/TEO
    tei = max(tei_low, tei_high)
    teo = min(teo_low, teo_high)

    # option : TSP location (low, center, high)
    tsp_center = []
    for low, high in zip(tsp_low, tsp_high):
        tsp_center.append(math.floor((low + high) / 2))
    if tsp_where=='low':
        tsp = tsp_low
    elif tsp_where=='high':
        tsp = tsp_high
    elif tsp_where=='center':
        tsp = tsp_center

    return tei, teo, tsp


def find_support(support_name, tei, teo, tsp, tsp_margin=75, tubeend_margin=150):
    """Support 구간 구하기 -> 1차 영역"""
    if support_name == 'TEI':
        support_start = tei + tubeend_margin
        support_end = tsp[0] - tsp_margin
    elif support_name == 'TEO':
        support_start = tsp[-1] + tsp_margin
        support_end = teo - tubeend_margin
    else:
        tsp_num = int(re.sub(r'[^0-9]', '', support_name))
        support_start = tsp[tsp_num-1] + tsp_margin
        if tsp_num >= len(tsp):
            support_end = teo - tubeend_margin
        else:
            support_end = tsp[tsp_num-1+1] - tsp_margin
    return support_start, support_end


def set_standard(channel):
    """결함 종류에 따른 검출 채널(detection_channel) 및 결함 그룹(defect_group) 설정"""
    if channel in ["IDI", "IDS", "DNG", "PVN"]:
        detection_channel = "CH1"
        defect_group = 1
    elif channel in ["DFI", "DFS"]:
        detection_channel = "CH3"
        defect_group = 1
    elif channel in ["DEP"]:
        detection_channel = "CH7"
        defect_group = 1
    elif channel in ["DSI", "DSS", "DNT"]:
    #     detection_channel = "Mix1(1-5)"
        detection_channel = "CH7"    
        defect_group = 2
    elif channel in ["WAR"]:
    #     detection_channel = "Mix2(2-6)"
        detection_channel = "CH7"
        defect_group = 2
    elif channel in ["WLL"]:
        detection_channel = "CH6"
        defect_group = 3
    else:
        detection_channel = None
        defect_group = None
    return detection_channel, defect_group


def find_defect(df, detection_channel, defect_group, support_start, support_end, tsp_margin=75, defect_margin=50):
    """결함 그룹 별 결함 구간 구하기 -> 2차 영역"""
    if defect_group == 1:  # IDI, IDS, DNG, DFI, DFS, DEP, PVN (Freespan 지역 결함)
        # 1. 해당 결함 검출용 채널의 Y값을 기준 값으로 설정
        detection_coord = detection_channel + "Y"
        # 2. Master 파일의 결함 위치(TSP) 활용하여 1차 영역 설정 (TSP ~ TSP +1)
        # -> find_support() 함수 사용
        # 3. 1차 영역 중 해당 채널 diff의 절댓값이 가장 큰 지점(Max Rate)을 중심으로 ±50point(or 100point) 2차 영역 설정
        diff_abs = df[detection_coord][support_start:support_end].diff().abs()
        diff_max = diff_abs[diff_abs == diff_abs.max()].index[0]
        defect_start = diff_max - defect_margin
        defect_end = diff_max + defect_margin
        # 4. 2차 영역에서 해당 채널 Y값의 최대, 최소 peak 값을 찾고 중심에서 두 peak의 평균값과 만나는 양방향 point까지 결함영역 설정(기존 방식 참조)
        # -> get_peak_wave_interval() 함수 사용
    elif defect_group == 2:  # DSI, DSS, DNT, WAR (Support 지역 결함)
        # 1. Ch7Y 값을 기준 값으로 설정
        detection_coord = "CH7Y"
        # 2. Master 파일의 결함 위치(TSP) 활용하여 1차 영역 설정 (TSP ~ TSP +1)
        # -> find_support() 함수 사용
        # 3. Master 파일의 결함 위치에 해당하는 TSP의 시작~끝 구간을 2차 영역으로 설정
        defect_start = support_start - 2*tsp_margin
        defect_end = support_start
    elif defect_group == 3:  # WLL (Freespan 체적 결함) - 데이터 점검 후 결정 예정
        # Master 파일의 결함 위치(TSP) 활용하여 결함 1차 영역 설정 (TSP ~ TSP +1)
        defect_start = support_start
        defect_end = support_end
    else:  # 기타 결함
        print("적합하지 않은 결함입니다.")
        return None, None
    return defect_start, defect_end


def get_lissajous(df, start, end, channel='CH3', size=1000, verbose=True):
    target_df = df[start:end]
    cond_ymax = target_df[channel+'Y']==target_df[channel+'Y'].max()
    cond_ymin = target_df[channel+'Y']==target_df[channel+'Y'].min()
    if verbose:
        print("검출 채널 :", channel)
        print("리사쥬 구간 :", (start, end))
#         print("x_ymax :", list(target_df[cond_ymax].index))
#         print("x_ymin :", list(target_df[cond_ymin].index))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            # name="lissajous_chart",
            x=target_df[channel+'X'],
            y=target_df[channel+'Y'],
            hovertext=target_df.index,
            mode='markers+lines',
            # marker=dict(size=5, line_width=0.5),
        )
    )
    fig.add_trace(
        go.Scattergl(
            name="ymax or ymin",
            x=target_df[channel+'X'][cond_ymax|cond_ymin],
            y=target_df[channel+'Y'][cond_ymax|cond_ymin],
            mode='markers',
            marker=dict(size=10, line_width=0, color='red'),
        )
    )
    fig.update_layout(
        title=f"defect:({start}, {end}) channel: {channel}"
    )
    fig.update_layout(
        width=size, height=size,
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    fig.show()
    
    return fig


def rename_old_columns(df):
    """컬럼명 변경 : 구버전 컬럼명(D1~4, A1~4) -> 신버전 컬럼명(CH1~CH8)"""
    if "D1Y" in df.columns:
        ch_cols = {
            "D1Y": "CH1Y",
            "D1X": "CH1X",
            "D2Y": "CH3Y",
            "D2X": "CH3X",
            "D3Y": "CH5Y",
            "D3X": "CH5X",
            "D4Y": "CH7Y",
            "D4X": "CH7X",
            "A1Y": "CH2Y",
            "A1X": "CH2X",
            "A2Y": "CH4Y",
            "A2X": "CH4X",
            "A3Y": "CH6Y",
            "A3X": "CH6X",
            "A4Y": "CH8Y",
            "A4X": "CH8X",
        }
        df = df.rename(columns=ch_cols)
        df = df[["CH1X", "CH1Y", "CH2X", "CH2Y", "CH3X", "CH3Y", "CH4X", "CH4Y", "CH5X", "CH5Y", "CH6X", "CH6Y", "CH7X", "CH7Y", "CH8X", "CH8Y"]]
    return df