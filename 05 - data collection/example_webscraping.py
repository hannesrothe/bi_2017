#!/usr/bin/env python3
 # -*- coding: utf-8 -*-

'''
@author: Hannes Rothe
@description: Fetch all institutions (universtities and companies) from Class central using a web scraper
@date: 2018-06-05
originally based on https://github.com/hannesrothe/berlin-startups
'''

import urllib

response = urllib.request.urlopen("https://www.class-central.com/universities")


from bs4 import BeautifulSoup

soup = BeautifulSoup(response, "html.parser")
instSoup = soup.findAll('a', 'university-name')

instData = []

import re

for inst in instSoup:
    print(inst)
    name = inst.text
    url = "https://www.class-central.com" + inst['href']
    crsNumber = re.findall('\d+', inst.find_parent('tr').find('span','courses-number').text)


#		if re.search('\%20', url) is None: #data cleansing for inconsistent url data
#			instData.append({'name': name, 'url': url,'type': instType, 'courses':crsNumber[0]})
#
#	response.close()


#==============================================================================
#    Export Pandas Table
import pandas as pd
#instDf = pd.DataFrame.from_dict(instData)
#instDf.index.name = "instId"
#
#instDf.to_csv("2017-institutions.csv", encoding="utf-8")


