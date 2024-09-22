# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:11:33 2022

@author: ludov
"""

import datetime as dt
import wrds
import psycopg2 
from dateutil.relativedelta import *
from pandas.tseries.offsets import *

# Connect to WRDS. Put Name and Password
conn=wrds.Connection() 


# This below dowloads from the server the data that we want #

crsp_m = conn.raw_sql("""
                      select a.permno, a.date, b.shrcd, b.exchcd,
                      a.ret, a.shrout, a.prc,a.retx
                      from crsp.msf as a
                      left join crsp.msenames as b
                      on a.permno=b.permno
                      and b.namedt<=a.date
                      and a.date<=b.nameendt
                      where a.date between '01/01/2006' and '12/31/2016'
                      and b.exchcd between 1 and 3
                      and b.shrcd between 10 and 12
                      """, date_cols=['date']) 

crsp_m.to_pickle('../../assets/data/crspm2005_2020.pkl')                      