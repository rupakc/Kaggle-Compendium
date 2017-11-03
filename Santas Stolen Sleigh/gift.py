# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 23:11:49 2016
Gift Object corressponding to the gifts which are to be delivered
@author: Rupak Chakraborty
"""
class Gift:
    
    def __init__(self,giftId,latitude,longitude,weight):
        self.giftId = giftId
        self.latitude = latitude
        self.longitude = longitude
        self.weight = weight

