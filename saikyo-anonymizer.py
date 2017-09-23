#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import shutil
import zipfile
import re
import os
import glob

import xml.etree.ElementTree as ET

anonymize_mode = True
debug_mode = False
preview_mode = False

work_dir= 'abc-anon-work'
preview_png = 'preview.png'
gif = 'image2.gif'
out_csv = 'strokes.csv'

argvs = sys.argv
if len(argvs) != 2 and len(argvs) != 6:
    print ('Usage: python %s filename [x1 y1 x2 y2]' % argvs[0])
    print ('       (x1 y1) position of left-top of deleting area')
    print ('       (x2 y2) position of right-bottom of deleting area')
    quit()

infilename = argvs[1]
name_auto_detect = False
x1=y1=x2=y2=-1

if len(argvs) == 2:
    name_auto_detect = True
else:
    x1 = int(argvs[2])
    y1 = int(argvs[3])
    x2 = int(argvs[4])
    y2 = int(argvs[5])


with zipfile.ZipFile(infilename, 'r') as zf:
    zf.extractall(work_dir)

    #tree = ET.parse(work_dir + '/customXml/item7.xml')
    tree = ET.parse(work_dir + "/customXml/" + max({i:os.path.getsize(work_dir + "/customXml/"+i) for i in os.listdir(work_dir + "/customXml/")}.items(),key=lambda x:x[1])[0])
    root = tree.getroot()

    if name_auto_detect:
        x1 = int(root.find('.//PointArea[@Type="AREA_NAME"]/X').text)
        y1 = int(root.find('.//PointArea[@Type="AREA_NAME"]/Y').text)    
        x2 = x1 + int(root.find('.//PointArea[@Type="AREA_NAME"]/Width').text)
        y2 = y1 + int(root.find('.//PointArea[@Type="AREA_NAME"]/Height').text)

    if preview_mode:
        width = int(root.find('.//PageSize/Width').text)
        height = int(root.find('.//PageSize/Height').text)
    
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (width, height), (255,255,255))
        imgdraw = ImageDraw.Draw(img)

    st = root.find('.//StrokeData')
    
#    print (st.text)
    lines = st.text.splitlines()
    output = ""
    
    for line in lines:
#        print(line)
        words = line.split(' ')
        
        # date/time
        date,time = words[0].split('-')

        mode = words[1]

        tool = words[2].translate(str.maketrans(';,',' ;'))

        output += date + ',' + time + ',' + mode + ',' + tool        

        if len(words) > 3:
            num1 = words[3]
            output += ',' + num1
            
        # stroke points
        if len(words) > 4:
            points = words[4].split(';')
            points_out = ""
            
            for point in points:
                x_hex,y_hex = point.split(',')
                x = int(x_hex, 16)
                y = int(y_hex, 16)

                if anonymize_mode:
                    if x >= x1 and y >= y1 and x <= x2 and y <= y2:
                        x = y = -1
                        
                points_out += str(x) + ':' + str(y) + ' '

                if preview_mode:
                    imgdraw.ellipse((x -1 , y -1 , x + 1,y + 1), fill=(0,0,0))
                    
            else:
                points_out = points_out[:-1]
            output += ',' + points_out

        if len(words) > 5:
            num2 = str(int(words[5],16))
            output += ',' + num2

        output += "\n"
        
#        print (output)

    out_dir = os.path.basename(infilename + "_out")
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if anonymize_mode:
        rect = "'rectangle " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "'"

        files = glob.glob(work_dir + "/media/image*.gif")
        for file in files:
            cmd = "convert " + file + " -fill black -draw " + rect + " '" + out_dir + "/" + os.path.basename(file) + "'"
            print (cmd)
            os.system(cmd)


    if preview_mode:
        img.save(out_dir + "/" + preview_png)

    f = open(out_dir + "/" + out_csv, 'w')
#    print(output)
    f.write(output)
    f.close()
    
    if (not debug_mode):
        shutil.rmtree(work_dir)

    

    
