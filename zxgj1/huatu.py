# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.pyplot import plot,figure,title,close,legend
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
import numpy as np

def jiegouhuatu1(wuch1,sdx=-1):

    fig=figure(1,figsize=(3.4,2.8),dpi=300) 
    if sdx<0:
        plt.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[0],
                    # c=wuch1[0],
                    # cmap='viridis_r',# 纵坐标  # 点的颜
                    label='$∆E_{1,j}$')
    else:
        plt.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[0],
                    # c=wuch1[0],
                    # cmap='viridis_r',# 纵坐标  # 点的颜
                    s=sdx,
                    label='$∆E_{1,j}$')
        
    
    plt.axhline(y=np.mean(wuch1[0,1:7]), color='b', linestyle=':',label='method 1')
    plt.axhline(y=1/2*(np.max(wuch1[0])+np.min(wuch1[0])), color='r', linestyle='--',label='method 2')
    plt.axhline(y=0.95*np.max(wuch1[0]), color='g', linestyle='-.',label='method 3')
    # plt.title('')
    plt.legend(fontsize=6.5)
    plt.xticks([1,2,3,4,5,6,7],fontsize=6.5)
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
    # ax.get_yaxis().get_offset_text().set(va='center', ha='left')
    offset_text = ax.yaxis.get_offset_text()
    # offset_text.set_position((0.05, -0.15))
    ax.yaxis.get_offset_text().set_fontsize(6.5)#设置1e6的大小与位置
    
    plt.yticks(fontsize=6.5)
    
    plt.xlabel('j',fontsize=6.5,labelpad=-1)#, ha='right', va='top'
    plt.ylabel('$∆E$',fontsize=6.5, labelpad=-1)#, va='top'ha='right'
    left_spine = plt.gca().spines['left']# 获取左边轴对象
    # left_spine.set_linewidth(0.1)# 设置线条粗细
    left_spine = plt.gca().spines['bottom']
    return fig

def jiegouhuatu2(wuch1):
    number_to_letter = {
        1: '(a)',
        2: '(b)',
        3: '(c)',
        4: '(d)',
        5: '(e)',
        6: '(f)'
    }

    fig=figure(2,figsize=(3.4,2.4),dpi=300)
    for ii in [2,3,4,5,6,7]:
        ax1=plt.subplot(2,3,ii-1)
        ax1.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[ii-1],s=2,label='$∆E_{i,j}$')
        quzz=np.delete(wuch1[ii-1],ii-1)
        ax1.axhline(y=np.mean(quzz), color='b', 
                    linestyle=':',label='method 1',linewidth=1)
        ax1.axhline(y=1/2*(np.max(wuch1[ii-1])+np.min(wuch1[ii-1])), color='r', 
                    linestyle='--',label='method 2',linewidth=1)
        ax1.axhline(y=0.95*np.max(wuch1[ii-1]), color='g', 
                    linestyle='-.',label='method 3',linewidth=1)
        
        ax1.set_xticks([1,2,3,4,5,6,7])
        
        ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        ax1.get_yaxis().get_offset_text().set(va='center', ha='left')
        ax1.yaxis.get_offset_text().set_fontsize(6.5)#设置1e6的大小与位置
        ax1.yaxis.get_offset_text().set_position((0,0.8))
        # ax1.set_xlabel('oscillator label',fontsize=6.5,labelpad=-1)#, ha='right', va='top'
        # ax1.set_ylabel('$∆E$',fontsize=6.5, labelpad=-1)#, va='to
        ax1.tick_params(axis='both', labelsize=6.5,pad=0,length=1)
        ax1.text(-0.2,1.05,number_to_letter[ii-1], fontsize=6.5,transform=ax1.transAxes)
    handles, labels = ax1.get_legend_handles_labels()

    fig.legend(
        handles,          # 图例句柄
        labels,           # 图例标签
        loc='lower center',  # 位置：图形上方居中
        # bbox_to_anchor=(0.9,0.1),  # 锚点坐标（相对图形）
        ncol=4,           # 列数（根据标签数量调整）
        fontsize=6.5,     # 字体大小
        frameon=False     # 去除边框
    )
    # plt.tight_layout()
    plt.subplots_adjust(
    top=0.913,
    bottom=0.144,
    left=0.09,
    right=0.956,
    hspace=0.319,
    wspace=0.292)
    return fig

def jiegouhuatu2duo(wuch1,wchsg):
    number_to_letter = {
        1: '(a)',
        2: '(b)',
        3: '(c)',
        4: '(d)',
        5: '(e)',
        6: '(f)'
    }

    fig=figure(2,figsize=(3.4,2.4),dpi=300)
    for ii in [2,3,4,5,6,7]:
        ax1=plt.subplot(2,3,ii-1)
        ax1.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[ii-1],s=2,label='$∆E_{i,j}$')
        quzz=np.delete(wuch1[ii-1],ii-1)
        ax1.axhline(y=np.mean(quzz), color='b', 
                    linestyle=':',label='method 1',linewidth=1)
        ax1.axhline(y=1/2*(np.max(wuch1[ii-1])+np.min(wuch1[ii-1])), color='r', 
                    linestyle='--',label='method 2',linewidth=1)
        ax1.axhline(y=0.95*np.max(wuch1[ii-1]), color='g', 
                    linestyle='-.',label='method 3',linewidth=1)
        
        ax1.axhline(y=wchsg[ii-1], color='y', 
                    linestyle='--',label='method 4',linewidth=1)
        ax1.set_xticks([1,2,3,4,5,6,7])
        
        ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        ax1.get_yaxis().get_offset_text().set(va='center', ha='left')
        ax1.yaxis.get_offset_text().set_fontsize(6.5)#设置1e6的大小与位置
        ax1.yaxis.get_offset_text().set_position((0,0.8))
        # ax1.set_xlabel('oscillator label',fontsize=6.5,labelpad=-1)#, ha='right', va='top'
        # ax1.set_ylabel('$∆E$',fontsize=6.5, labelpad=-1)#, va='to
        ax1.tick_params(axis='both', labelsize=6.5,pad=0,length=1)
        ax1.text(-0.2,1.05,number_to_letter[ii-1], fontsize=6.5,transform=ax1.transAxes)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,          # 图例句柄
        labels,           # 图例标签
        loc='lower center',  # 位置：图形上方居中
        # bbox_to_anchor=(0.1,0.1,0.9,0.1),  # 锚点坐标（相对图形）
        ncol=5,           # 列数（根据标签数量调整）
        fontsize=6.5,     # 字体大小
        handletextpad=0.1,# handle与text之间的距离
        handlelength=2,   #handle与legend边框的距离
        columnspacing=1,  #
        frameon=False     # 去除边框
    )
    # plt.tight_layout()
    plt.subplots_adjust(
    top=0.913,
    bottom=0.144,
    left=0.09,
    right=0.956,
    hspace=0.319,
    wspace=0.292)
    return fig

def jiegouhuatu21(wuch1):
    number_to_letter = {
        1: '(a)',
        2: '(b)',
        3: '(c)',
        4: '(d)',
        5: '(e)',
        6: '(f)'
    }
    fig=figure(3,figsize=(6.8,4.5),dpi=300)
    for ii in [2,3,4,5,6,7]:
        ax1=plt.subplot(2,3,ii-1)
        ax1.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[ii-1],s=2,label='$∆E_{i,j}$')
        quzz=np.delete(wuch1[ii-1],ii-1)
        ax1.axhline(y=np.mean(quzz), color='b', 
                    linestyle=':',label='method 1',linewidth=1)
        ax1.axhline(y=1/2*(np.max(wuch1[ii-1])+np.min(wuch1[ii-1])), color='r', 
                    linestyle='--',label='method 2',linewidth=1)
        ax1.axhline(y=0.95*np.max(wuch1[ii-1]), color='g', 
                    linestyle='-.',label='method 3',linewidth=1)
        
        ax1.set_xticks([1,2,3,4,5,6,7])
        
        ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        ax1.get_yaxis().get_offset_text().set(va='center', ha='left')
        ax1.yaxis.get_offset_text().set_fontsize(8.5)#设置1e6的大小与位置
        ax1.yaxis.get_offset_text().set_position((0,0.8))
        ax1.tick_params(axis='x', labelsize=8.5,pad=0,length=1)
        ax1.tick_params(axis='y', labelsize=8.5,pad=0,length=1)
        ax1.tick_params(axis='x', labelsize=8.5,pad=0,length=1)
        # ax1.set_title(number_to_letter[ii-1], fontsize=6.5,color='blue',transform=ax1.transAxes,bbox_to_anchor=(-0.12,1.02))
        ax1.text(-0.12,1.05,number_to_letter[ii-1], fontsize=8.5,transform=ax1.transAxes)#color='blue',
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(
        handles,          # 图例句柄
        labels,           # 图例标签
        loc='lower center',  # 位置：图形上方居中
        # bbox_to_anchor=(0.72,0.05),  # 锚点坐标（相对图形）
        ncol=4,           # 列数（根据标签数量调整）    
        fontsize=8.5,     # 字体大小
        frameon=False     # 去除边框
    )
    plt.subplots_adjust(top=0.928,
    bottom=0.103,
    left=0.064,
    right=0.968,
    hspace=0.202,
    wspace=0.179)
    return fig
def jiegouhuatu21duo(wuch1,wchsg):
    number_to_letter = {
        1: '(a)',
        2: '(b)',
        3: '(c)',
        4: '(d)',
        5: '(e)',
        6: '(f)'
    }
    fig=figure(3,figsize=(6.8,4.5),dpi=300)
    for ii in [2,3,4,5,6,7]:
        ax1=plt.subplot(2,3,ii-1)
        ax1.scatter([1,2,3,4,5,6,7],  # 横坐标
                    wuch1[ii-1],s=2,label='$∆E_{i,j}$')
        quzz=np.delete(wuch1[ii-1],ii-1)
        ax1.axhline(y=np.mean(quzz), color='b', 
                    linestyle=':',label='method 1',linewidth=1)
        ax1.axhline(y=1/2*(np.max(wuch1[ii-1])+np.min(wuch1[ii-1])), color='r', 
                    linestyle='--',label='method 2',linewidth=1)
        ax1.axhline(y=0.95*np.max(wuch1[ii-1]), color='g', 
                    linestyle='-.',label='method 3',linewidth=1)
        ax1.axhline(y=wchsg[ii-1], color='y', 
                    linestyle='--',label='method 4',linewidth=1)
        
        ax1.set_xticks([1,2,3,4,5,6,7])
        
        ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)
        ax1.get_yaxis().get_offset_text().set(va='center', ha='left')
        ax1.yaxis.get_offset_text().set_fontsize(8.5)#设置1e6的大小与位置
        ax1.yaxis.get_offset_text().set_position((0,0.8))
        ax1.tick_params(axis='x', labelsize=8.5,pad=0,length=1)
        ax1.tick_params(axis='y', labelsize=8.5,pad=0,length=1)
        ax1.tick_params(axis='x', labelsize=8.5,pad=0,length=1)
        # ax1.set_title(number_to_letter[ii-1], fontsize=6.5,color='blue',transform=ax1.transAxes,bbox_to_anchor=(-0.12,1.02))
        ax1.text(-0.12,1.05,number_to_letter[ii-1], fontsize=8.5,transform=ax1.transAxes)#color='blue',
    
    handles, labels = ax1.get_legend_handles_labels()
    
    fig.legend(
        handles,          # 图例句柄
        labels,           # 图例标签
        loc='lower center',  # 位置：图形上方居中
        # bbox_to_anchor=(0.72,0.05),  # 锚点坐标（相对图形）
        ncol=5,           # 列数（根据标签数量调整）    
        fontsize=8.5,     # 字体大小
        frameon=False     # 去除边框
    )
    plt.subplots_adjust(top=0.928,
    bottom=0.103,
    left=0.064,
    right=0.968,
    hspace=0.202,
    wspace=0.179)
    return fig


def 时序画图(ycx,y2,lwth=1,l1=200,l2=300):
    "大图片,不带标签"
    yx2=y2[:,::3]
    
    ycx1=ycx[0,:,::3]
    ycx2=ycx[1,:,::3]
    ycx3=ycx[2,:,::3]
    ycx4=ycx[3,:,::3]
    ycx5=ycx[4,:,::3]
    plt.rcParams.update({'font.size':6.5})
    # fig = plt.figure(dpi=300,figsize=(17/2.54,17/2.54))
    fig,ax=plt.subplots(6,1,dpi=300,figsize=(17/2.54,16/2.54))
    ax[0].plot(yx2[:l1,0],'r',label='actual',linewidth=lwth)
    ax[0].plot(ycx1[:l1,0],'b',label='rc',linewidth=lwth)
    ax[0].plot(ycx2[:l1,0],'g',label='I-rc',linewidth=lwth)
    ax[0].plot(ycx3[:l1,0],'c',label='P-rc',linewidth=lwth)
    ax[0].plot(ycx4[:l1,0],'k',label='v-rc',linewidth=lwth)
    ax[0].plot(ycx5[:l1,0],'y',label='C-rc',linewidth=lwth)

    # ax[0].set_xlabel('t',labelpad=0)
    ax[0].set_ylabel(r'$x_1(t)$',labelpad=-2.5)
    # ax1.set_title("(a),comparison",loc='left')
    ax[0].set_title("(a)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax[0].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0.2,handlelength=1)
    # 
    ax[1].plot(ycx1[:l2],linewidth=lwth)
    ax[2].plot(ycx2[:l2],linewidth=lwth)
    ax[3].plot(ycx3[:l2],linewidth=lwth)
    ax[4].plot(ycx4[:l2],linewidth=lwth)
    ax[5].plot(ycx5[:l2],linewidth=lwth)

    ax[1].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax[2].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax[3].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax[4].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax[5].tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)

    # ax2.set_xlabel('t',labelpad=0)
    # ax3.set_xlabel('t',labelpad=0)
    # ax5.set_xlabel('t',labelpad=0)
    ax[5].set_xlabel('t',labelpad=0)

    ax[1].set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax[2].set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax[2].set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax[4].set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax[5].set_ylabel(r'$x_i(t)$',labelpad=-2.5)

    ax[1].set_title("(b)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax[2].set_title("(c)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax[3].set_title("(d)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax[4].set_title("(e)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax[5].set_title("(f)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)

    # ax1.set_ylim(-15,15)
    # ax3.set_ylim(-20,20)
    # ax4.set_ylim(-20,20)

    plt.subplots_adjust(top=0.968,
    bottom=0.078,
    left=0.081,
    right=0.918,
    hspace=0.186,
    wspace=0.205)

    ax[1].legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax[2].legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax[3].legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax[4].legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax[5].legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    return fig,ax
def 时序画图_不一样大小(ycx,y2,lwth=1,l1=200,l2=300):
    "大图片,不一样大小不带标签"
    plt.rcParams.update({'font.size':6.5})
    yx2=y2[:,::3]
    ycx1=ycx[0,:,::3]
    ycx2=ycx[1,:,::3]
    ycx3=ycx[2,:,::3]
    ycx4=ycx[3,:,::3]
    ycx5=ycx[4,:,::3]
    fig = plt.figure(dpi=300,figsize=(17/2.54,17/2.54))
    # fig,ax=plt.subplots(6,1,dpi=300,figsize=(17/2.54,16/2.54))
    ax1=plt.subplot(7,1,(1,2))
    ax1.plot(yx2[:l1,0],'r',label='actual',linewidth=lwth)
    ax1.plot(ycx1[:l1,0],'b',label='rc',linewidth=lwth)
    ax1.plot(ycx2[:l1,0],'g',label='I-rc',linewidth=lwth)
    ax1.plot(ycx3[:l1,0],'c',label='P-rc',linewidth=lwth)
    ax1.plot(ycx4[:l1,0],'k',label='v-rc',linewidth=lwth)
    ax1.plot(ycx5[:l1,0],'y',label='C-rc',linewidth=lwth)

    # ax1.set_xlabel('t',labelpad=0)
    ax1.set_ylabel(r'$x_1(t)$',labelpad=-2.5)
    # ax1.set_title("(a),comparison",loc='left')
    ax1.set_title("(a)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax1.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0.2,handlelength=1)
    # 
    ax2=plt.subplot(7,1,3)
    ax3=plt.subplot(7,1,4)
    ax4=plt.subplot(7,1,5)
    ax5=plt.subplot(7,1,6)
    ax6=plt.subplot(7,1,7)
    ax2.plot(ycx1[:l2],linewidth=lwth)
    ax3.plot(ycx2[:l2],linewidth=lwth)
    ax4.plot(ycx3[:l2],linewidth=lwth)
    ax5.plot(ycx4[:l2],linewidth=lwth)
    ax6.plot(ycx5[:l2],linewidth=lwth)

    ax2.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax3.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax4.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax5.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax6.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)

    # ax2.set_xlabel('t',labelpad=0)
    # ax3.set_xlabel('t',labelpad=0)
    # ax5.set_xlabel('t',labelpad=0)
    ax6.set_xlabel('t',labelpad=0)

    ax2.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax3.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax4.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax5.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax6.set_ylabel(r'$x_i(t)$',labelpad=-2.5)

    ax2.set_title("(b)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax3.set_title("(c)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax4.set_title("(d)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax5.set_title("(e)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)
    ax6.set_title("(f)",loc='left',fontsize=7.5,pad=0,x=-0.05,y=0.95)

    # ax1.set_ylim(-15,15)
    ax3.set_ylim(-20,20)
    ax4.set_ylim(-20,20)

    plt.subplots_adjust(top=0.968,
    bottom=0.078,
    left=0.081,
    right=0.918,
    hspace=0.186,
    wspace=0.205)

    ax2.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax3.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax4.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax5.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    ax6.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.2,
            labelspacing=0,handlelength=1)
    return fig    
    # ax2.set_title("(b),rc prediction",loc='left')
    # ax3.set_title("(c),I-rc prediction",loc='left')
    # ax4.set_title("(d),C-rc prediction",loc='left')
    # ax5.set_title("(d),C-rc prediction",loc='left')
    # ax6.set_title("(d),C-rc prediction",loc='left')    
def 时序画图小_不一样大小(ycx,y2,lwth=0.5,l1=200,l2=300):
    plt.rcParams.update({'font.size':6.5})
    yx2=y2[:,::3]
    ycx1=ycx[0,:,::3]
    ycx2=ycx[1,:,::3]
    ycx3=ycx[2,:,::3]
    ycx4=ycx[3,:,::3]
    ycx5=ycx[4,:,::3]
    fig = plt.figure(dpi=300,figsize=(8.5/2.54,9/2.54))
    ax1=plt.subplot(7,1,(1,2))
    ax1.plot(yx2[:200,0],'r',label='actual',linewidth=lwth)
    ax1.plot(ycx1[:200,0],'b',label='rc',linewidth=lwth)
    ax1.plot(ycx2[:200,0],'g',label='I-rc',linewidth=lwth)
    ax1.plot(ycx3[:200,0],'c',label='P-rc',linewidth=lwth)
    ax1.plot(ycx4[:200,0],'k',label='v-rc',linewidth=lwth)
    ax1.plot(ycx5[:200,0],'y',label='C-rc',linewidth=lwth)

    ax1.set_ylabel(r'$x_1(t)$',labelpad=-2.5)
    ax1.set_title("(a)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax1.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.5,
            labelspacing=0.,handlelength=1)

    # 
    ax2=plt.subplot(7,1,3)
    ax3=plt.subplot(7,1,4)
    ax4=plt.subplot(7,1,5)
    ax5=plt.subplot(7,1,6)
    ax6=plt.subplot(7,1,7)
    ax2.plot(ycx1[:400],linewidth=lwth)
    ax3.plot(ycx2[:400],linewidth=lwth)
    ax4.plot(ycx3[:400],linewidth=lwth)
    ax5.plot(ycx4[:400],linewidth=lwth)
    ax6.plot(ycx5[:400],linewidth=lwth)

    ax1.set_ylim(-20,20)
    ax2.set_ylim(-22,22)
    ax3.set_ylim(-22,22)
    ax4.set_ylim(-22,22)


    ax6.set_xlabel('t',labelpad=0)


    ax2.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax3.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax4.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax5.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax6.set_ylabel(r'$x_i(t)$',labelpad=-2.5)

    ax2.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax3.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax4.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax5.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax6.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)

    ax2.set_title("(b)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    # ax2.set_title("(b),rc prediction",loc='left')
    ax3.set_title("(c)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    # ax3.set_title("(c),I-rc prediction",loc='left')
    # ax4.set_title("(d),C-rc prediction",loc='left')
    ax4.set_title("(d)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax5.set_title("(e)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax6.set_title("(f)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)

    # ax2.set_xticklabels([])
    # ax3.set_xticklabels([])



    # ax2.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
    #            loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
    #            handletextpad=0.5,borderaxespad=0.2,
    #            labelspacing=0,handlelength=1)
    ax4.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.5,
            labelspacing=2.4,handlelength=1)
    # ax4.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
    #            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
    #            handletextpad=0.5,borderaxespad=0.2,
    #            labelspacing=0,handlelength=1)


    # plt.tight_layout()

    plt.subplots_adjust(top=0.95,
    bottom=0.1,
    left=0.13,
    right=0.837,
    hspace=0.38,
    wspace=0.2)
    return fig
def 时序画图小(ycx,y2,lwth=0.5,l1=200,l2=300):
    plt.rcParams.update({'font.size':6.5})
    yx2=y2[:,::3]
    ycx1=ycx[0,:,::3]
    ycx2=ycx[1,:,::3]
    ycx3=ycx[2,:,::3]
    ycx4=ycx[3,:,::3]
    ycx5=ycx[4,:,::3]
    fig = plt.figure(dpi=300,figsize=(8.5/2.54,9/2.54))
    lwth=0.5
    ax1=plt.subplot(6,1,1)
    ax1.plot(yx2[:200,0],'r',label='actual',linewidth=lwth)
    ax1.plot(ycx1[:200,0],'b',label='rc',linewidth=lwth)
    ax1.plot(ycx2[:200,0],'g',label='I-rc',linewidth=lwth)
    ax1.plot(ycx3[:200,0],'c',label='P-rc',linewidth=lwth)
    ax1.plot(ycx4[:200,0],'k',label='v-rc',linewidth=lwth)
    ax1.plot(ycx5[:200,0],'y',label='C-rc',linewidth=lwth)


    ax1.set_ylabel(r'$x_1(t)$',labelpad=-2.5)
    ax1.set_title("(a)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax1.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.5,
            labelspacing=0.,handlelength=1)

    # 
    ax2=plt.subplot(6,1,2)
    ax3=plt.subplot(6,1,3)
    ax4=plt.subplot(6,1,4)
    ax5=plt.subplot(6,1,5)
    ax6=plt.subplot(6,1,6)
    ax2.plot(ycx1[:400],linewidth=lwth)
    ax3.plot(ycx2[:400],linewidth=lwth)
    ax4.plot(ycx3[:400],linewidth=lwth)
    ax5.plot(ycx4[:400],linewidth=lwth)
    ax6.plot(ycx5[:400],linewidth=lwth)

    ax1.set_ylim(-20,20)
    ax2.set_ylim(-22,22)
    ax3.set_ylim(-22,22)
    ax4.set_ylim(-22,22)


    ax6.set_xlabel('t',labelpad=0)


    ax2.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax3.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax4.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax5.set_ylabel(r'$x_i(t)$',labelpad=-2.5)
    ax6.set_ylabel(r'$x_i(t)$',labelpad=-2.5)

    ax2.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax3.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax4.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax5.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)
    ax6.tick_params(axis='both', labelsize=6.5,pad=0.2,length=1)

    ax2.set_title("(b)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    # ax2.set_title("(b),rc prediction",loc='left')
    ax3.set_title("(c)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    # ax3.set_title("(c),I-rc prediction",loc='left')
    # ax4.set_title("(d),C-rc prediction",loc='left')
    ax4.set_title("(d)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax5.set_title("(e)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)
    ax6.set_title("(f)",loc='left',fontsize=7.5,pad=0,x=-0.15,y=0.92)

    # ax2.set_xticklabels([])
    # ax3.set_xticklabels([])



    # ax2.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
    #            loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
    #            handletextpad=0.5,borderaxespad=0.2,
    #            labelspacing=0,handlelength=1)
    ax4.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
            loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 6},
            handletextpad=0.5,borderaxespad=0.5,
            labelspacing=3,handlelength=1)
    # ax4.legend(labels=(r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$'),
    #            loc='upper left', bbox_to_anchor=(1, 1),prop={'size': 6},
    #            handletextpad=0.5,borderaxespad=0.2,
    #            labelspacing=0,handlelength=1)


    # plt.tight_layout()

    plt.subplots_adjust(top=0.95,
    bottom=0.1,
    left=0.13,
    right=0.837,
    hspace=0.38,
    wspace=0.2)
    return fig
def 误差画图(yche,yy2,llsh=200,vmax1=50):
    wucha1=np.abs(yche[0,:5000,:]-yy2)
    wucha2=np.abs(yche[1,:5000,:]-yy2)
    wucha3=np.abs(yche[2,:5000,:]-yy2)
    wucha4=np.abs(yche[3,:5000,:]-yy2)
    wucha5=np.abs(yche[4,:5000,:]-yy2)
    l1,l2=wucha1.shape
    plt.rcParams.update({'font.size':6.5})
    # plt.imshow()
    fig, axs = plt.subplots(5,1,dpi=300,figsize=(17/2.54,11/2.54))
    im = axs[0].imshow(wucha1[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im,ax=axs[0],pad=0.05)

    im = axs[1].imshow(wucha2[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5,llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im, ax=axs[1],pad=0.05)
    im = axs[2].imshow(wucha3[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[2],pad=0.05)
    im = axs[3].imshow(wucha4[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[3],pad=0.05)
    im = axs[4].imshow(wucha5[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    labelpad1=-0
    axs[4].set_xlabel('t',labelpad=0)
    for ii in range(5):
        axs[ii].set_ylabel(r'dim.',labelpad=labelpad1)
        if ii!=4:
            axs[ii].set_xticks([])
        axs[ii].tick_params(axis='both', labelsize=6.5,pad=0.2,length=2)
        # axs[ii].axis('off')
    axs[0].set_title("(a)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[1].set_title("(b)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[2].set_title("(c)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[3].set_title("(d)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[4].set_title("(e)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    plt.subplots_adjust(top=0.967,
    bottom=0.113,
    left=0.083,
    right=0.954,
    hspace=0.05,
    wspace=0.2)
    # plt.tight_layout()
    cbar = fig.colorbar(im,ax=axs,pad=0.02)
    axs[0].text(2,l2-0.15*l2,'RC',color='purple',fontsize=7.5)
    axs[1].text(2,l2-0.15*l2,'I-RC',color='purple',fontsize=7.5)
    axs[2].text(2,l2-0.15*l2,'P-RC',color='purple',fontsize=7.5)
    axs[3].text(2,l2-0.15*l2,'R-RC',color='purple',fontsize=7.5)
    axs[4].text(2,l2-0.15*l2,'C-RC',color='purple',fontsize=7.5)
    return fig,axs

def 误差画图_带小图1(yche,yy2,llsh=200,vmax1=50):
    "小图带坐标"
    wucha1=np.abs(yche[0,:5000,:]-yy2)
    wucha2=np.abs(yche[1,:5000,:]-yy2)
    wucha3=np.abs(yche[2,:5000,:]-yy2)
    wucha4=np.abs(yche[3,:5000,:]-yy2)
    wucha5=np.abs(yche[4,:5000,:]-yy2)
    l1,l2=wucha1.shape
    plt.rcParams.update({'font.size':6.5})
    # plt.imshow()
    fig, axs = plt.subplots(5,1,dpi=300,figsize=(17/2.54,11/2.54))
    im = axs[0].imshow(wucha1[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im,ax=axs[0],pad=0.05)

    im = axs[1].imshow(wucha2[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5,llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im, ax=axs[1],pad=0.05)
    im = axs[2].imshow(wucha3[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[2],pad=0.05)
    im = axs[3].imshow(wucha4[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[3],pad=0.05)
    im = axs[4].imshow(wucha5[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    labelpad1=-0
    axs[4].set_xlabel('t',labelpad=0)
    for ii in range(5):
        axs[ii].set_ylabel(r'dim.',labelpad=labelpad1)
        if ii!=4:
            axs[ii].set_xticks([])
        axs[ii].tick_params(axis='both', labelsize=6.5,pad=0.2,length=2)
        # axs[ii].axis('off')
    axs[0].set_title("(a)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[1].set_title("(b)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[2].set_title("(c)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[3].set_title("(d)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[4].set_title("(e)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    plt.subplots_adjust(top=0.967,
    bottom=0.113,
    left=0.083,
    right=0.954,
    hspace=0.05,
    wspace=0.2)
    # plt.tight_layout()
    cbar = fig.colorbar(im,ax=axs,pad=0.02)
    axs[0].text(2,l2-0.15*l2,'RC',color='purple',fontsize=7.5)
    axs[1].text(2,l2-0.15*l2,'I-RC',color='purple',fontsize=7.5)
    axs[2].text(2,l2-0.15*l2,'P-RC',color='purple',fontsize=7.5)
    axs[3].text(2,l2-0.15*l2,'R-RC',color='purple',fontsize=7.5)
    axs[4].text(2,l2-0.15*l2,'C-RC',color='purple',fontsize=7.5)
    axins = axs[0].inset_axes((0.8, 0.7, 0.1, 0.25))
    im = axins.imshow(wucha1[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5 ])
    axins.set_xlim(150.5, 154.5)
    axins.set_ylim(10.5, 12.5)
    axins.tick_params(axis='both', labelsize=6.5,pad=0.2,length=2)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
    return fig,axs
def 误差画图_带小图2(yche,yy2,llsh=200,vmax1=50):
    "小图不带坐标"
    wucha1=np.abs(yche[0,:5000,:]-yy2)
    wucha2=np.abs(yche[1,:5000,:]-yy2)
    wucha3=np.abs(yche[2,:5000,:]-yy2)
    wucha4=np.abs(yche[3,:5000,:]-yy2)
    wucha5=np.abs(yche[4,:5000,:]-yy2)
    l1,l2=wucha1.shape
    plt.rcParams.update({'font.size':6.5})
    # plt.imshow()
    fig, axs = plt.subplots(5,1,dpi=300,figsize=(17/2.54,11/2.54))
    im = axs[0].imshow(wucha1[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im,ax=axs[0],pad=0.05)

    im = axs[1].imshow(wucha2[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5,llsh+0.5,0.5, l2+0.5 ])
    # cbar = ax1.figure.colorbar(im, ax=axs[1],pad=0.05)
    im = axs[2].imshow(wucha3[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[2],pad=0.05)
    im = axs[3].imshow(wucha4[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    # cbar = ax1.figure.colorbar(im, ax=axs[3],pad=0.05)
    im = axs[4].imshow(wucha5[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5  ])
    labelpad1=-0
    axs[4].set_xlabel('t',labelpad=0)
    for ii in range(5):
        axs[ii].set_ylabel(r'dim.',labelpad=labelpad1)
        if ii!=4:
            axs[ii].set_xticks([])
        axs[ii].tick_params(axis='both', labelsize=6.5,pad=0.2,length=2)
        # axs[ii].axis('off')
    axs[0].set_title("(a)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[1].set_title("(b)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[2].set_title("(c)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[3].set_title("(d)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    axs[4].set_title("(e)",loc='left',va='baseline', fontsize=7.5, pad=0.04,x=-0.06,y=0.94)
    plt.subplots_adjust(top=0.967,
    bottom=0.113,
    left=0.083,
    right=0.954,
    hspace=0.05,
    wspace=0.2)
    # plt.tight_layout()
    cbar = fig.colorbar(im,ax=axs,pad=0.02)
    axs[0].text(2,l2-0.15*l2,'RC',color='purple',fontsize=7.5)
    axs[1].text(2,l2-0.15*l2,'I-RC',color='purple',fontsize=7.5)
    axs[2].text(2,l2-0.15*l2,'P-RC',color='purple',fontsize=7.5)
    axs[3].text(2,l2-0.15*l2,'R-RC',color='purple',fontsize=7.5)
    axs[4].text(2,l2-0.15*l2,'C-RC',color='purple',fontsize=7.5)
    axins = axs[0].inset_axes((0.8, 0.7, 0.1, 0.25))
    im = axins.imshow(wucha1[:llsh,:].T, cmap='viridis_r',vmax=vmax1,origin='lower',aspect="auto",
                       extent=[0.5, llsh+0.5,0.5, l2+0.5 ])
    axins.set_xlim(150.5, 154.5)
    axins.set_ylim(10.5, 12.5)
    axins.tick_params(axis='both', labelsize=6.5,pad=0.2,length=2)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    mark_inset(axs[0], axins, loc1=2, loc2=4, fc="none", ec='k', lw=1)
    return fig,axs










