import os
import cv2
import numpy as np
import pandas as pd
import argparse
from scipy.signal import resample
from scipy.signal import find_peaks

from perspective_transform import perspective_transform

def read_img(f):
	im = perspective_transform(f)
	warped = im.copy()
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = cv2.resize(im, (2200, 1600))
	_, im = cv2.threshold(im, 128, 255, cv2.THRESH_OTSU)
	kernel = np.ones((3, 3), np.uint8)
	im = cv2.erode(im, kernel) 
	return im, warped

def histogram(arr, axis=0):
	hist = []
	if axis == 0:
		for i in range(arr.shape[axis]):
			hist.append(len(np.where(arr[i, :]==0)[0]))
	if axis == 1:
		for i in range(arr.shape[axis]):
			hist.append(len(np.where(arr[:, i]==0)[0]))
	return hist

def find_baseline(img):
	vh = histogram(img[400:, 100:], 0)
	vh = np.array(vh)
	peaks = find_peaks(vh, distance=100)[0]

	temp = vh[peaks]
	[bl0, bl1, bl2, bl3] = np.sort(np.array(peaks[np.argsort(temp)[-4:]]))
	return [bl0+400, bl1+400, bl2+400, bl3+400]

def noise_removement(ll):
	for i in range(1, ll.shape[0]-1):
		for j in range(1, ll.shape[1]-1):
			arr = ll[i-1:i+2, j-1:j+2]
			if len(np.where(arr==0)[0]) <= 1:
				ll[i, j] = 255
	return ll

def detact_leads(img, bl):
	gap = []
	for i in range(len(bl)-1):
		gap.append(bl[i+1]-bl[i])
	gap = np.array(gap).mean().astype('int')
	gap = gap - gap//8

	hh = histogram(img[bl[1]-gap:bl[1]+gap, 50:img.shape[1]-50], 1)
	st, en = 0, len(hh)
	for i in range(len(hh)):
		if hh[i] > 3:
			st = i
			break
	for i in range(len(hh)-1, 0, -1):
		if hh[i] > 3:
			en = i
			break
	img = img[:, 50+st:50+en]
	N = 50
	L = img.shape[1]//4
	l0 = img[bl[0]-gap:bl[0]+gap, N:L]
	l1 = img[bl[1]-gap:bl[1]+gap, N:L]
	l2 = img[bl[2]-gap:bl[2]+gap, N:L]

	l3 = img[bl[0]-gap:bl[0]+gap, N+L:2*L]
	l4 = img[bl[1]-gap:bl[1]+gap, N+L:2*L]
	l5 = img[bl[2]-gap:bl[2]+gap, N+L:2*L]

	l6 = img[bl[0]-gap:bl[0]+gap, N+2*L:3*L]
	l7 = img[bl[1]-gap:bl[1]+gap, N+2*L:3*L]
	l8 = img[bl[2]-gap:bl[2]+gap, N+2*L:3*L]

	l9 = img[bl[0]-gap:bl[0]+gap, N+3*L:4*L]
	l10 = img[bl[1]-gap:bl[1]+gap, N+3*L:4*L]
	l11 = img[bl[2]-gap:bl[2]+gap, N+3*L:4*L]

	l12 = img[bl[3]-gap:bl[3]+gap, :]

	return l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12

def remove_redundant_lead(im, gap=40):
	bl = np.argmax(histogram(im, 0))
	im_ = 255*np.ones_like(im)
	for r in range(3):
		pt, cnt = 0, 0
		while True:
			x = np.random.randint(im.shape[1])
			if im[bl,x] == 0:
				pt = x
				break
			cnt += 1
			if cnt == 10000:
				break
			
		col_ = np.where(im[:, pt]==0)[0]
		pos= []
		for i in range(len(col_)):
			if col_[i] < bl-15 or col_[i] > bl+15:
				pos.append(i)
		col_ = np.delete(col_, pos)
		im_[col_, pt] = 0

		amax, amin = max(col_), min(col_)
		for i in range(pt, im.shape[1]):
			if amin - gap > 0 and amax + gap < im.shape[0]:
				mask = np.where(im[amin-gap:amax+gap, i]==0)[0] + amin-gap
			elif amin - gap < 0 and amax + gap < im.shape[0]:
				mask = np.where(im[0:amax+gap, i]==0)[0]
			elif amin - gap < 0 and amax + gap > im.shape[0]:
				mask = np.where(im[amin-gap:im.shape[0], i]==0)[0] + amin-gap
			else:
				mask = np.where(im[0:im.shape[0], i]==0)[0]
			im_[mask, i] = 0
			try:
				amax, amin = max(mask), min(mask)
			except:
				pass

		amax, amin = max(col_), min(col_)
		for i in range(pt-1, -1, -1):
			if amin - gap > 0 and amax + gap < im.shape[0]:
				mask = np.where(im[amin-gap:amax+gap, i]==0)[0] + amin-gap
			elif amin - gap < 0 and amax + gap < im.shape[0]:
				mask = np.where(im[0:amax+gap, i]==0)[0]
			elif amin - gap > 0 and amax + gap > im.shape[0]:
				mask = np.where(im[amin-gap:im.shape[0], i]==0)[0] + amin-gap
			else:
				mask = np.where(im[0:im.shape[0], i]==0)[0]

			im_[mask, i] = 0
			try:
				amax, amin = max(mask), min(mask)
			except:
				pass
		return im_ 

def digitization0(im, w=40, h=80, fs=20):
	im = im[:, 20:im.shape[1]-20]
	hist = histogram(im, 0)
	bl = np.argmax(hist)

	sig = []
	for i in range(im.shape[1]):
		col = im[:, i]
		if len(np.where(col==0)[0]) != 0:
			mag = np.mean(np.where(col==0)[0])
		else:
			mag = bl
		sig.append(mag)
	sig = (np.array(sig) - bl)/h

	n_samples = int(fs*im.shape[1]/w)
	sig = resample(sig, n_samples)
	return 1-sig

def digitization(f):
	im, warped = read_img(f)
	bl = find_baseline(im)
	w, h = 40, 80

	l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12 = detact_leads(im, bl)

	ll0 = remove_redundant_lead(l0, gap=20); ll0 = cv2.medianBlur(ll0, 3); sig0 = digitization0(ll0, w, h)
	ll1 = remove_redundant_lead(l1, gap=20); ll1 = cv2.medianBlur(ll1, 3); sig1 = digitization0(ll1, w, h)
	ll2 = remove_redundant_lead(l2, gap=20); ll2 = cv2.medianBlur(ll2, 3); sig2 = digitization0(ll2, w, h)
	ll3 = remove_redundant_lead(l3, gap=20); ll3 = cv2.medianBlur(ll3, 3); sig3 = digitization0(ll3, w, h)
	ll4 = remove_redundant_lead(l4, gap=20); ll4 = cv2.medianBlur(ll4, 3); sig4 = digitization0(ll4, w, h)
	ll5 = remove_redundant_lead(l5, gap=20); ll5 = cv2.medianBlur(ll5, 3); sig5 = digitization0(ll5, w, h)
	ll6 = remove_redundant_lead(l6, gap=20); ll6 = cv2.medianBlur(ll6, 3); sig6 = digitization0(ll6, w, h)
	ll7 = remove_redundant_lead(l7, gap=20); ll7 = cv2.medianBlur(ll7, 3); sig7 = digitization0(ll7, w, h)
	ll8 = remove_redundant_lead(l8, gap=20); ll8 = cv2.medianBlur(ll8, 3); sig8 = digitization0(ll8, w, h)
	ll9 = remove_redundant_lead(l9, gap=20); ll9 = cv2.medianBlur(ll9, 3); sig9 = digitization0(ll9, w, h)
	ll10 = remove_redundant_lead(l10, gap=20); ll10 = cv2.medianBlur(ll10, 3); sig10 = digitization0(ll10, w, h)
	ll11 = remove_redundant_lead(l11, gap=20); ll11 = cv2.medianBlur(ll11, 3); sig11 = digitization0(ll11, w, h)
	ll12 = remove_redundant_lead(l12, gap=20); ll12 = cv2.medianBlur(ll12, 3); sig12 = digitization0(ll12, w, h)

	N = len(sig0)
	sig = np.zeros((12, N))
	sig[0] = sig0; sig[1] = sig1; sig[2] = sig2; sig[3] = sig3; sig[4] = sig4; sig[5] = sig5; 
	sig[6] = sig6; sig[7] = sig7; sig[8] = sig8; sig[9] = sig9; sig[10] = sig10; sig[11] = sig11

	return warped, sig, sig12

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Path to raw image, saved image, and saved signal.')
	parser.add_argument('--imagepath', help='path to raw image')
	parser.add_argument('--savepath', help='path to saved image')
	parser.add_argument('--signalpath', help='path to saved siganl')
	args = parser.parse_args()
	imagepath = args.imagepath
	savepath = args.savepath
	signalpath = args.signalpath
	warped, sig, _ = digitization(imagepath)
	cv2.imwrite(savepath, warped)

	df = pd.DataFrame(columns=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
	df['I'] = sig[0]
	df['II'] = sig[1]
	df['III'] = sig[2]
	df['aVR'] = sig[3]
	df['aVL'] = sig[4]
	df['aVF'] = sig[5]
	df['V1'] = sig[6]
	df['V2'] = sig[7]
	df['V3'] = sig[8]
	df['V4'] = sig[9]
	df['V5'] = sig[10]
	df['V6'] = sig[11]
	df.to_csv(signalpath, index=None)

	print('[INFO] Perspective-transformed image was saved at ' ,os.path.join(os.getcwd(), savepath))
	print('[INFO] Extracted signals were saved at ' ,os.path.join(os.getcwd(), signalpath))