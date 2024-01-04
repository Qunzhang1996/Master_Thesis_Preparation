import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",  # 使用衬线字体/主字体
    "text.usetex": False,  # 使用内嵌数学公式
    "pgf.rcfonts": False,  # 不从rc参数设置字体
    "pgf.preamble": "\n".join([
        "\\usepackage{units}",  # 加载额外的包
        "\\usepackage{metalogo}",
        "\\usepackage{unicode-math}",  # unicode数学设置
        r"\usepackage{amsfonts}",
        r"\usepackage{amssymb}",
        r"\setmathfont{xits-math.otf}",
        r"\usepackage{amsmath}",
        r"\setmainfont{DejaVu Serif}",  # 通过preamble设置衬线字体
        r"\usepackage{siunitx}",  # 需要正立的\micro符号
        r"\sisetup{detect-all}",  # 强制siunitx实际使用你的字体
        r"\usepackage{helvet}",  # 设置正常字体
        r"\usepackage{sansmath}",  # 加载sansmath使数学公式->helvet
        r"\sansmath"  # <- 诡异！需要告诉tex使用这个！
    ])
})
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

TITLE_SIZE = 16
LEGEND_SIZE = 16
TICK_SIZE = 14
AXIS_TITLE = 20
AXIS_LABEL = 20
FONT_SIZE = TITLE_SIZE
plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=AXIS_TITLE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_LABEL)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})