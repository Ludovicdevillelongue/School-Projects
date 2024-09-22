# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:43:50 2020

@author: ludov
"""

'''---------------------------// Scrap bank notes from ECB //-------------------------------'''

import pandasdmx as sdmx

import pandas as pd

ecb = sdmx.Request('ECB')

#http connection
ecb_via_proxy = sdmx.Request(
 'ECB',
proxies={'http': 'http://1.2.3.4:5678'}
)
ecb_via_proxy.session.proxies
{'http': 'http://1.2.3.4:5678'}


#dataflow information

flow_msg = ecb.dataflow()


#URL linked to request

flow_msg.response.url

#answer to header request
flow_msg.response.headers

#dataflow and metadata collection
print('\n\n',flow_msg, '\n\n')

#dataflow conversion in pandas series
dataflows = sdmx.to_pandas(flow_msg.dataflow)
print('\n\n',dataflows)

#verification that dataflow are saved in dataflow serie
len(dataflows)

#found appropriate dataflow
print('\n\n', dataflows[dataflows.str.contains('government statistics', case=False)])

#extract metadata
exr_msg = ecb.dataflow('BKN')
exr_msg.response.url
exr_flow = exr_msg.dataflow.BKN

#show data's structuration definition
montrer la definition de structuration de la data
dsd = exr_flow.structure
print('\n\n',dsd)

#explore definition components 
print('\n\n',dsd.dimensions.components)
print('\n\n',dsd.attributes.components)
print('\n\n',dsd.measures.components)

#Codelist that contains valid values for this dimension in the data flow:
cl = dsd.dimensions.get('FREQ').local_representation.enumerated
print('\n\n',cl)

#convert codelist in pandas data
convertir la codelist en data pandas
datapd= sdmx.to_pandas(cl)
print(datapd)

#constraints
print('\n\n nombre de codelist',len(exr_msg.codelist.CL_BKN_TYPE))
exr_msg.constraint.BKN_CONSTRAINTS
cr = exr_msg.constraint.BKN_CONSTRAINTS.data_content_region[0]
c1 = sdmx.to_pandas(cr.member['BKN_TYPE'].values)
print('\n\n membre valides' ,len(c1))
c2 = sdmx.to_pandas(cr.member['BKN_DENOM'].values)
print('\n\n membres valides',len(c2))
#c2 members only
print(c2 - c1)
#c1 members only
print(c1 - c2)





#create a key for request
key = dict(BKN_DENOM=['50N2','10N2'])
#beginning of data period
params=dict(Startperiod="2016")

#extraction in generic format
ecb = sdmx.Request('ECB')
data_msg = ecb.data('BKN', key=key, params=params)
data_msg.response.headers['content-type']



data = data_msg.data[0]

#data type
type(data)

#number of series in data
print(len(data.series))

#series selection
print(list(data.series.keys())[5])

set(series_key.FREQ for series_key in data.series.keys())
daily = [s for sk, s in data.series.items() if sk.FREQ == 'M']
cur_df = pd.concat(sdmx.to_pandas(daily)).unstack()



#series conversion in dataframe
df2 = sdmx.to_pandas(
data,
datetime=dict(dim='TIME_PERIOD', freq='FREQ', axis=1))
df2.columns
print(df2)
