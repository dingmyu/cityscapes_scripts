import os

Dir = 'exp/cityscapes/psp50_dist_16_713_fine_4/result/epoch_480/val/ss/gray/'
import numpy as np
import cv2
flist = os.listdir(Dir)
for index, item in enumerate(flist):
	if index%20 == 0:
		print(index)
	pic = cv2.imread(Dir + item, cv2.IMREAD_GRAYSCALE)
	pic[pic==18]=33
	pic[pic==17]=32
	pic[pic==16]=31
	pic[pic==15]=28
	pic[pic==14]=27
	pic[pic==13]=26
	pic[pic==12]=25
	pic[pic==11]=24
	pic[pic==10]=23
	pic[pic==9]=22
	pic[pic==8]=21
	pic[pic==7]=20
	pic[pic==6]=19
	pic[pic==5]=17
	pic[pic==4]=13
	pic[pic==3]=12
	pic[pic==2]=11
	pic[pic==1]=8
	pic[pic==0]=7
	pic[pic==255]=0
	pic = np.uint8(pic)
	cv2.imwrite('submit_4/' + item, pic)
