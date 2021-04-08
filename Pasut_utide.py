# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import tkinter as tk

import utide

def date_parser(year, month, day, hour):
    year, month, day, hour = map(int, (year, month, day, hour))
    return datetime.datetime(year, month, day, hour)
def td():

    fl0=nr0.get()
    slat=nr1.get()
    tzone=nr2.get()
    mtd=nr3.get()

    lat=float(slat)
    tzone=int(tzone)
    print('method={}'.format(mtd))
    from sklearn import metrics
    from math import sqrt
    UtdF='../data/'+fl0[0:-4]+'.dtf'
    fli=open('../data/'+fl0,'r',encoding='cp1252')
    flo=open(UtdF,'w')
    i=1
    for ln in fli:
        if ln[0].isnumeric():
            fl1=ln.split(' ')
            fl2=fl1[0].split('/')
            fl3=fl1[1].split(':')
            rval=float(fl1[2])/100
            cday = '%2d' % int(fl2[0])
            cmonth = '%2d' % int(fl2[1])
            cyear = '%4d' % int(fl2[2])
            chour = '%2d.0000' % int(fl3[0])
            buf = '%6d'% i +' '+cyear+' '+cmonth+' '+cday+' '+chour+' %7.4f' % rval+' 0\n'
            flo.write(buf)
            i=i+1
    fli.close()
    flo.close()

    # Names of the columns that will be used to make a "datetime" column:
    parse_dates = dict(datetime=['year', 'month', 'day','hour'])

    # Names of the original columns in the file, including only
    # the ones we will use; we are skipping the first, which appears
    # to be seconds from the beginning.
    names = ['year', 'month', 'day', 'hour', 'elev', 'flag']

    obs = pd.read_table(UtdF,
                        sep=' ',
                        names=names,
                        skipinitialspace=True,
                        #delim_whitespace=True,
                        index_col='datetime',
                        usecols=range(1, 7),
                        na_values='999.999',
                        parse_dates=parse_dates,
                        date_parser=date_parser,
                       )
    bad = obs['flag'] == 2
    corrected = obs['flag'] == 1

    obs.loc[bad, 'elev'] = np.nan
    Mobs=obs['elev'].mean()
    obs['anomaly'] = obs['elev'] - Mobs
    obs['anomaly'] = obs['anomaly'].interpolate() + Mobs
    print('{} points were flagged "bad" and interpolated'.format(bad.sum()))
    print('{} points were flagged "corrected" and left unchanged'.format(corrected.sum()))

    time = mdates.date2num(obs.index.to_pydatetime())-tzone/24

    coef = utide.solve(time, obs['anomaly'].values,
                       lat=lat,
                       method=mtd,
                       conf_int='MC')
    #print(coef.keys())
    tide = utide.reconstruct(time, coef)
    #print(tide.keys())

    print('\n')
    flo=open('../out/'+UtdF[8:-4]+'_'+mtd+'.coe','w')

    ig=len(coef['name'])
    print('{:4s} {:^8s} {:^8s} {:^10s} {:^10s}'.format('Coef','A','A_ci','g','g_ci'))
    flo.write('{:4s} {:^8s} {:^8s} {:^10s} {:^10s} \n'.format('Coef','A','A_ci','g','g_ci'))
    for i in range(ig):
        print('{:4s} {:8.5f} {:8.5f} {:10.5f} {:10.5f}'.format(coef['name'][i],coef['A'][i],coef['A_ci'][i],coef['g'][i],coef['g_ci'][i]))
        flo.write('{:4s} {:8.5f} {:8.5f} {:10.5f} {:10.5f}\n'.format(coef['name'][i],coef['A'][i],coef['A_ci'][i],coef['g'][i],coef['g_ci'][i]))
    print('\n\n')
    #t = obs.index.values  # dtype is '<M8[ns]' (numpy datetime64)
    # It is more efficient to supply the time directly as matplotlib
    # datenum floats:

    t = tide.t_mpl
    res=obs.anomaly - tide.h
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharey=True, sharex=True)
    ax0.plot(t, obs.anomaly, label=u'Observations', color='C0')
    ax1.plot(t, tide.h, label=u'Tide Fit', color='C1')
    ax2.plot(t, res, label=u'Residual', color='C2')
    ax2.xaxis_date()
    fig.legend(ncol=3, loc='lower center')
    fig.autofmt_xdate()
    fig.suptitle('Comparison observation data with UTide in Station ' +UtdF[8:-4], fontsize=16)
    fig.savefig('../out/'+UtdF[8:-4]+'_'+mtd+'.png')

    print('Std Dev= {:6.3f}'.format(res.std()))

    mse=metrics.mean_squared_error(np.array(obs.anomaly) , tide.h)

    print('rmse   = {:6.3f}'.format(sqrt(mse)))
    print('\n\n')
    a=datetime.date(2019,1,1).toordinal()-719163.0
    b=datetime.date(2020,1,1).toordinal()-719163.0
    times=np.arange(float(a),float(b),1/24)

    tides = utide.reconstruct(times, coef)
    t = tides.t_mpl

    fig, (ax0) = plt.subplots(nrows=1, sharey=True, sharex=True)
    ax0.plot(t, tides.h, label=u'Tide Prediction', color='C1')
    ax0.xaxis_date()
    fig.autofmt_xdate()
    fig.suptitle('Tide Prediction with UTide in Station ' +UtdF[8:-4], fontsize=16)
    fig.savefig('../out/'+UtdF[8:-4]+'_'+mtd+'_P.png')

    print('Minimum= {:6.3f}'.format(tide.h.min()))
    print('Maximum= {:6.3f}'.format(tide.h.max()))
    print('Mean   = {:6.3f}'.format(tide.h.mean()))
    print('\n\n')
    print('Minimum= {:6.3f}'.format(tides.h.min()))
    print('Maximum= {:6.3f}'.format(tides.h.max()))
    print('Mean   = {:6.3f}'.format(tides.h.mean()))
    print('rmse   = {:6.3f}'.format(sqrt(mse)))
    flo.write('\n\n')
    flo.write('Std Dev= {:6.3f}\n'.format(res.std()))
    flo.write('rmse   = {:6.3f}\n'.format(sqrt(mse)))
    flo.write('\n\n')
    flo.write('Minimum= {:6.3f}\n'.format(tide.h.min()))
    flo.write('Maximum= {:6.3f}\n'.format(tide.h.max()))
    flo.write('Mean   = {:6.3f}\n'.format(tide.h.mean()))
    flo.write('Minimum= {:6.3f}\n'.format(tides.h.min()))
    flo.write('Maximum= {:6.3f}\n'.format(tides.h.max()))
    flo.write('Mean   = {:6.3f}\n'.format(tides.h.mean()))
    flo.close()

    print('\n\nFinished\n')

def iUtd():
    for widget in rt.winfo_children():
        widget.destroy()

    lf=tk.LabelFrame(rt,text='UTIDE Version = '+utide.__version__,relief='raised')
    lf.pack()
    tk.Label(lf,text='Input    =',font=("Mono",10)).grid(row=0,column=0)
    gr0=tk.Entry(lf,textvariable=nr0)
    gr0.delete(0, tk.END)
    gr0.insert(0,'Maileppet.txt')
    gr0.grid(row=0,column=1)
    tk.Label(lf,text='Latitude =',font=("Mono",10)).grid(row=1,column=0)
    gr1=tk.Entry(lf,textvariable=nr1)
    gr1.delete(0, tk.END)
    gr1.insert(0,'-1.563597222')
    gr1.grid(row=1,column=1)
    tk.Label(lf,text='Time Zone=',font=("Mono",10)).grid(row=2,column=0)
    gr2=tk.Entry(lf,textvariable=nr2)
    gr2.delete(0, tk.END)
    gr2.insert(0,'7')
    gr2.grid(row=2,column=1)
    tk.Label(lf,text='Method   =',font=("Mono",10)).grid(row=3,column=0)
    opt1 = tk.OptionMenu(lf, nr3, 'robust', 'ols')
    nr3.set('robust')
    opt1.config(width=15)
    opt1.grid(row=3,column=1)


    tk.Button(lf,text='Run',command=lambda:td()).grid(row=12,column=1)
    tk.Button(rt,text='Quit',command=rt.quit).pack(side=tk.BOTTOM)

rt=tk.Tk()
rt.geometry("300x185")
nr0=tk.StringVar()
nr1=tk.StringVar()
nr2=tk.StringVar()
nr3=tk.StringVar()
rt.title('Tide Calculation (UTide) ')
iUtd()
rt.mainloop()