import cv2
import numpy as np
import webcolors
import math

def landmarks_to_np(landmarks, dtype="int"):
    # landmarks
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_centers(img, landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=-1)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    # pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    # cv2.polylines(img, [pts], False, (255,0,0), 1) 
    # cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    # cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def get_aligned_face(img, left, right):
    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)
    scale = desired_dist / dist 
    angle = np.degrees(np.arctan2(dy,dx)) 
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
    
    return aligned_face

def judge_eyeglass(img):
    img = cv2.GaussianBlur(img, (11,11), 0) 

    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) 
    sobel_y = cv2.convertScaleAbs(sobel_y) 
    # cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y 
    
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] 
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1*0.3 + measure_2*0.7
   
    # Determine the discriminant value based on the relationship 
    # between the evaluation value and the threshold
    if measure > 0.21:
        judge = True
    else:
        judge = False
    # print(judge)
    return judge

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    if h > sh or w > sw: # shirking image
        interp = cv2.INTER_AREA
    else:   # streching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h 
    saspect = sw/sh

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def rgb_to_hsv(r, g, b):
    r = float(r)
    g = float(g)
    b = float(b)
    high = max(r, g, b)
    low = min(r, g, b)
    h, s, v = high, high, high

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, v

def hsv_to_rgb(h, s, v):
    i = math.floor(h*6)
    f = h*6 - i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)

    r, g, b = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ][int(i%6)]

    return r, g, b

class ColorNames:
    # Src: http://www.w3schools.com/html/html_colornames.asp  
    WebColorMap = {}
    WebColorMap["AliceBlue"] = "#F0F8FF"
    WebColorMap["AntiqueWhite"] = "#FAEBD7"
    WebColorMap["Aqua"] = "#00FFFF"
    WebColorMap["Aquamarine"] = "#7FFFD4"
    WebColorMap["Azure"] = "#F0FFFF"
    WebColorMap["Beige"] = "#F5F5DC"
    WebColorMap["Bisque"] = "#FFE4C4"
    WebColorMap["Black"] = "#000000"
    WebColorMap["BlanchedAlmond"] = "#FFEBCD"
    WebColorMap["Blue"] = "#0000FF"
    WebColorMap["BlueViolet"] = "#8A2BE2"
    WebColorMap["Brown"] = "#A52A2A"
    WebColorMap["BurlyWood"] = "#DEB887"
    WebColorMap["CadetBlue"] = "#5F9EA0"
    WebColorMap["Chartreuse"] = "#7FFF00"
    WebColorMap["Chocolate"] = "#D2691E"
    WebColorMap["Coral"] = "#FF7F50"
    WebColorMap["CornflowerBlue"] = "#6495ED"
    WebColorMap["Cornsilk"] = "#FFF8DC"
    WebColorMap["Crimson"] = "#DC143C"
    WebColorMap["Cyan"] = "#00FFFF"
    WebColorMap["DarkBlue"] = "#00008B"
    WebColorMap["DarkCyan"] = "#008B8B"
    WebColorMap["DarkGoldenRod"] = "#B8860B"
    WebColorMap["DarkGray"] = "#A9A9A9"
    WebColorMap["DarkGrey"] = "#A9A9A9"
    WebColorMap["DarkGreen"] = "#006400"
    WebColorMap["DarkKhaki"] = "#BDB76B"
    WebColorMap["DarkMagenta"] = "#8B008B"
    WebColorMap["DarkOliveGreen"] = "#556B2F"
    WebColorMap["Darkorange"] = "#FF8C00"
    WebColorMap["DarkOrchid"] = "#9932CC"
    WebColorMap["DarkRed"] = "#8B0000"
    WebColorMap["DarkSalmon"] = "#E9967A"
    WebColorMap["DarkSeaGreen"] = "#8FBC8F"
    WebColorMap["DarkSlateBlue"] = "#483D8B"
    WebColorMap["DarkSlateGray"] = "#2F4F4F"
    WebColorMap["DarkSlateGrey"] = "#2F4F4F"
    WebColorMap["DarkTurquoise"] = "#00CED1"
    WebColorMap["DarkViolet"] = "#9400D3"
    WebColorMap["DeepPink"] = "#FF1493"
    WebColorMap["DeepSkyBlue"] = "#00BFFF"
    WebColorMap["DimGray"] = "#696969"
    WebColorMap["DimGrey"] = "#696969"
    WebColorMap["DodgerBlue"] = "#1E90FF"
    WebColorMap["FireBrick"] = "#B22222"
    WebColorMap["FloralWhite"] = "#FFFAF0"
    WebColorMap["ForestGreen"] = "#228B22"
    WebColorMap["Fuchsia"] = "#FF00FF"
    WebColorMap["Gainsboro"] = "#DCDCDC"
    WebColorMap["GhostWhite"] = "#F8F8FF"
    WebColorMap["Gold"] = "#FFD700"
    WebColorMap["GoldenRod"] = "#DAA520"
    WebColorMap["Gray"] = "#808080"
    WebColorMap["Grey"] = "#808080"
    WebColorMap["Green"] = "#008000"
    WebColorMap["GreenYellow"] = "#ADFF2F"
    WebColorMap["HoneyDew"] = "#F0FFF0"
    WebColorMap["HotPink"] = "#FF69B4"
    WebColorMap["red"] = "#CD5C5C"
    WebColorMap["Indigo"] = "#4B0082"
    WebColorMap["Ivory"] = "#FFFFF0"
    WebColorMap["Khaki"] = "#F0E68C"
    WebColorMap["Lavender"] = "#E6E6FA"
    WebColorMap["LavenderBlush"] = "#FFF0F5"
    WebColorMap["LawnGreen"] = "#7CFC00"
    WebColorMap["LemonChiffon"] = "#FFFACD"
    WebColorMap["LightBlue"] = "#ADD8E6"
    WebColorMap["LightCoral"] = "#F08080"
    WebColorMap["LightCyan"] = "#E0FFFF"
    WebColorMap["LightGoldenRodYellow"] = "#FAFAD2"
    WebColorMap["LightGray"] = "#D3D3D3"
    WebColorMap["LightGrey"] = "#D3D3D3"
    WebColorMap["LightGreen"] = "#90EE90"
    WebColorMap["LightPink"] = "#FFB6C1"
    WebColorMap["LightSalmon"] = "#FFA07A"
    WebColorMap["LightSeaGreen"] = "#20B2AA"
    WebColorMap["LightSkyBlue"] = "#87CEFA"
    WebColorMap["LightSlateGray"] = "#778899"
    WebColorMap["LightSlateGrey"] = "#778899"
    WebColorMap["LightSteelBlue"] = "#B0C4DE"
    WebColorMap["LightYellow"] = "#FFFFE0"
    WebColorMap["Lime"] = "#00FF00"
    WebColorMap["LimeGreen"] = "#32CD32"
    WebColorMap["Linen"] = "#FAF0E6"
    WebColorMap["Magenta"] = "#FF00FF"
    WebColorMap["Maroon"] = "#800000"
    WebColorMap["AquaMarine"] = "#66CDAA"
    WebColorMap["Blue"] = "#0000CD"
    WebColorMap["Orchid"] = "#BA55D3"
    WebColorMap["Purple"] = "#9370D8"
    WebColorMap["SeaGreen"] = "#3CB371"
    WebColorMap["SlateBlue"] = "#7B68EE"
    WebColorMap["SpringGreen"] = "#00FA9A"
    WebColorMap["Turquoise"] = "#48D1CC"
    WebColorMap["VioletRed"] = "#C71585"
    WebColorMap["MidnightBlue"] = "#191970"
    WebColorMap["MintCream"] = "#F5FFFA"
    WebColorMap["MistyRose"] = "#FFE4E1"
    WebColorMap["Cream"] = "#FFE4B5"
    WebColorMap["Sandy"] = "#FFDEAD"
    WebColorMap["Navy"] = "#000080"
    WebColorMap["OldLace"] = "#FDF5E6"
    WebColorMap["Olive"] = "#808000"
    WebColorMap["OliveDrab"] = "#6B8E23"
    WebColorMap["Orange"] = "#FFA500"
    WebColorMap["OrangeRed"] = "#FF4500"
    WebColorMap["Orchid"] = "#DA70D6"
    WebColorMap["PaleGoldenRod"] = "#EEE8AA"
    WebColorMap["PaleGreen"] = "#98FB98"
    WebColorMap["PaleTurquoise"] = "#AFEEEE"
    WebColorMap["PaleVioletRed"] = "#D87093"
    WebColorMap["PapayaWhip"] = "#FFEFD5"
    WebColorMap["PeachPuff"] = "#FFDAB9"
    WebColorMap["BurntOrange"] = "#CD853F"
    WebColorMap["Pink"] = "#FFC0CB"
    WebColorMap["Plum"] = "#DDA0DD"
    WebColorMap["PowderBlue"] = "#B0E0E6"
    WebColorMap["Purple"] = "#800080"
    WebColorMap["Red"] = "#FF0000"
    WebColorMap["RosyBrown"] = "#BC8F8F"
    WebColorMap["RoyalBlue"] = "#4169E1"
    WebColorMap["SaddleBrown"] = "#8B4513"
    WebColorMap["Salmon"] = "#FA8072"
    WebColorMap["SandyBrown"] = "#F4A460"
    WebColorMap["SeaGreen"] = "#2E8B57"
    WebColorMap["SeaShell"] = "#FFF5EE"
    WebColorMap["Sienna"] = "#A0522D"
    WebColorMap["Silver"] = "#C0C0C0"
    WebColorMap["SkyBlue"] = "#87CEEB"
    WebColorMap["SlateBlue"] = "#6A5ACD"
    WebColorMap["SlateGray"] = "#708090"
    WebColorMap["SlateGrey"] = "#708090"
    WebColorMap["Snow"] = "#FFFAFA"
    WebColorMap["SpringGreen"] = "#00FF7F"
    WebColorMap["SteelBlue"] = "#4682B4"
    WebColorMap["Tan"] = "#D2B48C"
    WebColorMap["Teal"] = "#008080"
    WebColorMap["Thistle"] = "#D8BFD8"
    WebColorMap["Tomato"] = "#FF6347"
    WebColorMap["Turquoise"] = "#40E0D0"
    WebColorMap["Violet"] = "#EE82EE"
    WebColorMap["Wheat"] = "#F5DEB3"
    WebColorMap["White"] = "#FFFFFF"
    WebColorMap["WhiteSmoke"] = "#F5F5F5"
    WebColorMap["Yellow"] = "#FFFF00"
    WebColorMap["YellowGreen"] = "#9ACD32"
    
    # src: http://www.imagemagick.org/script/color.php
    ImageMagickColorMap = {}
    ImageMagickColorMap["snow"] = "#FFFAFA"
    ImageMagickColorMap["snow1"] = "#FFFAFA"
    ImageMagickColorMap["snow2"] = "#EEE9E9"
    ImageMagickColorMap["RosyBrown1"] = "#FFC1C1"
    ImageMagickColorMap["RosyBrown2"] = "#EEB4B4"
    ImageMagickColorMap["snow3"] = "#CDC9C9"
    ImageMagickColorMap["LightCoral"] = "#F08080"
    ImageMagickColorMap["Red"] = "#FF6A6A"
    ImageMagickColorMap["RosyBrown3"] = "#CD9B9B"
    ImageMagickColorMap["Red"] = "#EE6363"
    ImageMagickColorMap["RosyBrown"] = "#BC8F8F"
    ImageMagickColorMap["brown1"] = "#FF4040"
    ImageMagickColorMap["firebrick1"] = "#FF3030"
    ImageMagickColorMap["brown2"] = "#EE3B3B"
    ImageMagickColorMap["Red"] = "#CD5C5C"
    ImageMagickColorMap["Red"] = "#CD5555"
    ImageMagickColorMap["firebrick2"] = "#EE2C2C"
    ImageMagickColorMap["snow4"] = "#8B8989"
    ImageMagickColorMap["brown3"] = "#CD3333"
    ImageMagickColorMap["red"] = "#FF0000"
    ImageMagickColorMap["red1"] = "#FF0000"
    ImageMagickColorMap["RosyBrown4"] = "#8B6969"
    ImageMagickColorMap["firebrick3"] = "#CD2626"
    ImageMagickColorMap["red2"] = "#EE0000"
    ImageMagickColorMap["firebrick"] = "#B22222"
    ImageMagickColorMap["brown"] = "#A52A2A"
    ImageMagickColorMap["red3"] = "#CD0000"
    ImageMagickColorMap["Red"] = "#8B3A3A"
    ImageMagickColorMap["brown4"] = "#8B2323"
    ImageMagickColorMap["firebrick4"] = "#8B1A1A"
    ImageMagickColorMap["DarkRed"] = "#8B0000"
    ImageMagickColorMap["red4"] = "#8B0000"
    ImageMagickColorMap["maroon"] = "#800000"
    ImageMagickColorMap["LightPink1"] = "#FFAEB9"
    ImageMagickColorMap["LightPink3"] = "#CD8C95"
    ImageMagickColorMap["LightPink4"] = "#8B5F65"
    ImageMagickColorMap["LightPink2"] = "#EEA2AD"
    ImageMagickColorMap["LightPink"] = "#FFB6C1"
    ImageMagickColorMap["pink"] = "#FFC0CB"
    ImageMagickColorMap["crimson"] = "#DC143C"
    ImageMagickColorMap["pink1"] = "#FFB5C5"
    ImageMagickColorMap["pink2"] = "#EEA9B8"
    ImageMagickColorMap["pink3"] = "#CD919E"
    ImageMagickColorMap["pink4"] = "#8B636C"
    ImageMagickColorMap["PaleVioletRed4"] = "#8B475D"
    ImageMagickColorMap["PaleVioletRed"] = "#DB7093"
    ImageMagickColorMap["PaleVioletRed2"] = "#EE799F"
    ImageMagickColorMap["PaleVioletRed1"] = "#FF82AB"
    ImageMagickColorMap["PaleVioletRed3"] = "#CD6889"
    ImageMagickColorMap["LavenderBlush"] = "#FFF0F5"
    ImageMagickColorMap["LavenderBlush1"] = "#FFF0F5"
    ImageMagickColorMap["LavenderBlush3"] = "#CDC1C5"
    ImageMagickColorMap["LavenderBlush2"] = "#EEE0E5"
    ImageMagickColorMap["LavenderBlush4"] = "#8B8386"
    ImageMagickColorMap["maroon"] = "#B03060"
    ImageMagickColorMap["HotPink3"] = "#CD6090"
    ImageMagickColorMap["VioletRed3"] = "#CD3278"
    ImageMagickColorMap["VioletRed1"] = "#FF3E96"
    ImageMagickColorMap["VioletRed2"] = "#EE3A8C"
    ImageMagickColorMap["VioletRed4"] = "#8B2252"
    ImageMagickColorMap["HotPink2"] = "#EE6AA7"
    ImageMagickColorMap["HotPink1"] = "#FF6EB4"
    ImageMagickColorMap["HotPink4"] = "#8B3A62"
    ImageMagickColorMap["HotPink"] = "#FF69B4"
    ImageMagickColorMap["DeepPink"] = "#FF1493"
    ImageMagickColorMap["DeepPink1"] = "#FF1493"
    ImageMagickColorMap["DeepPink2"] = "#EE1289"
    ImageMagickColorMap["DeepPink3"] = "#CD1076"
    ImageMagickColorMap["DeepPink4"] = "#8B0A50"
    ImageMagickColorMap["maroon1"] = "#FF34B3"
    ImageMagickColorMap["maroon2"] = "#EE30A7"
    ImageMagickColorMap["maroon3"] = "#CD2990"
    ImageMagickColorMap["maroon4"] = "#8B1C62"
    ImageMagickColorMap["VioletRed"] = "#C71585"
    ImageMagickColorMap["VioletRed"] = "#D02090"
    ImageMagickColorMap["orchid2"] = "#EE7AE9"
    ImageMagickColorMap["orchid"] = "#DA70D6"
    ImageMagickColorMap["orchid1"] = "#FF83FA"
    ImageMagickColorMap["orchid3"] = "#CD69C9"
    ImageMagickColorMap["orchid4"] = "#8B4789"
    ImageMagickColorMap["thistle1"] = "#FFE1FF"
    ImageMagickColorMap["thistle2"] = "#EED2EE"
    ImageMagickColorMap["plum1"] = "#FFBBFF"
    ImageMagickColorMap["plum2"] = "#EEAEEE"
    ImageMagickColorMap["thistle"] = "#D8BFD8"
    ImageMagickColorMap["thistle3"] = "#CDB5CD"
    ImageMagickColorMap["plum"] = "#DDA0DD"
    ImageMagickColorMap["violet"] = "#EE82EE"
    ImageMagickColorMap["plum3"] = "#CD96CD"
    ImageMagickColorMap["thistle4"] = "#8B7B8B"
    ImageMagickColorMap["fuchsia"] = "#FF00FF"
    ImageMagickColorMap["magenta"] = "#FF00FF"
    ImageMagickColorMap["magenta1"] = "#FF00FF"
    ImageMagickColorMap["plum4"] = "#8B668B"
    ImageMagickColorMap["magenta2"] = "#EE00EE"
    ImageMagickColorMap["magenta3"] = "#CD00CD"
    ImageMagickColorMap["DarkMagenta"] = "#8B008B"
    ImageMagickColorMap["magenta4"] = "#8B008B"
    ImageMagickColorMap["purple"] = "#800080"
    ImageMagickColorMap["Orchid"] = "#BA55D3"
    ImageMagickColorMap["Orchid1"] = "#E066FF"
    ImageMagickColorMap["Orchid2"] = "#D15FEE"
    ImageMagickColorMap["Orchid3"] = "#B452CD"
    ImageMagickColorMap["Orchid4"] = "#7A378B"
    ImageMagickColorMap["DarkViolet"] = "#9400D3"
    ImageMagickColorMap["DarkOrchid"] = "#9932CC"
    ImageMagickColorMap["DarkOrchid1"] = "#BF3EFF"
    ImageMagickColorMap["DarkOrchid3"] = "#9A32CD"
    ImageMagickColorMap["DarkOrchid2"] = "#B23AEE"
    ImageMagickColorMap["DarkOrchid4"] = "#68228B"
    ImageMagickColorMap["purple"] = "#A020F0"
    ImageMagickColorMap["indigo"] = "#4B0082"
    ImageMagickColorMap["BlueViolet"] = "#8A2BE2"
    ImageMagickColorMap["purple2"] = "#912CEE"
    ImageMagickColorMap["purple3"] = "#7D26CD"
    ImageMagickColorMap["purple4"] = "#551A8B"
    ImageMagickColorMap["purple1"] = "#9B30FF"
    ImageMagickColorMap["Purple"] = "#9370DB"
    ImageMagickColorMap["Purple1"] = "#AB82FF"
    ImageMagickColorMap["Purple2"] = "#9F79EE"
    ImageMagickColorMap["Purple3"] = "#8968CD"
    ImageMagickColorMap["Purple4"] = "#5D478B"
    ImageMagickColorMap["DarkSlateBlue"] = "#483D8B"
    ImageMagickColorMap["LightSlateBlue"] = "#8470FF"
    ImageMagickColorMap["SlateBlue"] = "#7B68EE"
    ImageMagickColorMap["SlateBlue"] = "#6A5ACD"
    ImageMagickColorMap["SlateBlue1"] = "#836FFF"
    ImageMagickColorMap["SlateBlue2"] = "#7A67EE"
    ImageMagickColorMap["SlateBlue3"] = "#6959CD"
    ImageMagickColorMap["SlateBlue4"] = "#473C8B"
    ImageMagickColorMap["GhostWhite"] = "#F8F8FF"
    ImageMagickColorMap["lavender"] = "#E6E6FA"
    ImageMagickColorMap["blue"] = "#0000FF"
    ImageMagickColorMap["blue1"] = "#0000FF"
    ImageMagickColorMap["blue2"] = "#0000EE"
    ImageMagickColorMap["blue3"] = "#0000CD"
    ImageMagickColorMap["Blue"] = "#0000CD"
    ImageMagickColorMap["blue4"] = "#00008B"
    ImageMagickColorMap["DarkBlue"] = "#00008B"
    ImageMagickColorMap["MidnightBlue"] = "#191970"
    ImageMagickColorMap["navy"] = "#000080"
    ImageMagickColorMap["NavyBlue"] = "#000080"
    ImageMagickColorMap["RoyalBlue"] = "#4169E1"
    ImageMagickColorMap["RoyalBlue1"] = "#4876FF"
    ImageMagickColorMap["RoyalBlue2"] = "#436EEE"
    ImageMagickColorMap["RoyalBlue3"] = "#3A5FCD"
    ImageMagickColorMap["RoyalBlue4"] = "#27408B"
    ImageMagickColorMap["CornflowerBlue"] = "#6495ED"
    ImageMagickColorMap["LightSteelBlue"] = "#B0C4DE"
    ImageMagickColorMap["LightSteelBlue1"] = "#CAE1FF"
    ImageMagickColorMap["LightSteelBlue2"] = "#BCD2EE"
    ImageMagickColorMap["LightSteelBlue3"] = "#A2B5CD"
    ImageMagickColorMap["LightSteelBlue4"] = "#6E7B8B"
    ImageMagickColorMap["SlateGray4"] = "#6C7B8B"
    ImageMagickColorMap["SlateGray1"] = "#C6E2FF"
    ImageMagickColorMap["SlateGray2"] = "#B9D3EE"
    ImageMagickColorMap["SlateGray3"] = "#9FB6CD"
    ImageMagickColorMap["LightSlateGray"] = "#778899"
    ImageMagickColorMap["LightSlateGrey"] = "#778899"
    ImageMagickColorMap["SlateGray"] = "#708090"
    ImageMagickColorMap["SlateGrey"] = "#708090"
    ImageMagickColorMap["DodgerBlue"] = "#1E90FF"
    ImageMagickColorMap["DodgerBlue1"] = "#1E90FF"
    ImageMagickColorMap["DodgerBlue2"] = "#1C86EE"
    ImageMagickColorMap["DodgerBlue4"] = "#104E8B"
    ImageMagickColorMap["DodgerBlue3"] = "#1874CD"
    ImageMagickColorMap["LightBlue"] = "#F0F8FF"
    ImageMagickColorMap["SteelBlue4"] = "#36648B"
    ImageMagickColorMap["SteelBlue"] = "#4682B4"
    ImageMagickColorMap["SteelBlue1"] = "#63B8FF"
    ImageMagickColorMap["SteelBlue2"] = "#5CACEE"
    ImageMagickColorMap["SteelBlue3"] = "#4F94CD"
    ImageMagickColorMap["SkyBlue4"] = "#4A708B"
    ImageMagickColorMap["SkyBlue1"] = "#87CEFF"
    ImageMagickColorMap["SkyBlue2"] = "#7EC0EE"
    ImageMagickColorMap["SkyBlue3"] = "#6CA6CD"
    ImageMagickColorMap["LightSkyBlue"] = "#87CEFA"
    ImageMagickColorMap["LightSkyBlue4"] = "#607B8B"
    ImageMagickColorMap["LightSkyBlue1"] = "#B0E2FF"
    ImageMagickColorMap["LightSkyBlue2"] = "#A4D3EE"
    ImageMagickColorMap["LightSkyBlue3"] = "#8DB6CD"
    ImageMagickColorMap["SkyBlue"] = "#87CEEB"
    ImageMagickColorMap["LightBlue3"] = "#9AC0CD"
    ImageMagickColorMap["DeepSkyBlue"] = "#00BFFF"
    ImageMagickColorMap["DeepSkyBlue1"] = "#00BFFF"
    ImageMagickColorMap["DeepSkyBlue2"] = "#00B2EE"
    ImageMagickColorMap["DeepSkyBlue4"] = "#00688B"
    ImageMagickColorMap["DeepSkyBlue3"] = "#009ACD"
    ImageMagickColorMap["LightBlue1"] = "#BFEFFF"
    ImageMagickColorMap["LightBlue2"] = "#B2DFEE"
    ImageMagickColorMap["LightBlue"] = "#ADD8E6"
    ImageMagickColorMap["LightBlue4"] = "#68838B"
    ImageMagickColorMap["PowderBlue"] = "#B0E0E6"
    ImageMagickColorMap["CadetBlue1"] = "#98F5FF"
    ImageMagickColorMap["CadetBlue2"] = "#8EE5EE"
    ImageMagickColorMap["CadetBlue3"] = "#7AC5CD"
    ImageMagickColorMap["CadetBlue4"] = "#53868B"
    ImageMagickColorMap["turquoise1"] = "#00F5FF"
    ImageMagickColorMap["turquoise2"] = "#00E5EE"
    ImageMagickColorMap["turquoise3"] = "#00C5CD"
    ImageMagickColorMap["turquoise4"] = "#00868B"
    ImageMagickColorMap["CadetBlue"] = "#5F9EA0"
    ImageMagickColorMap["DarkBlue"] = "#030e2b"
    ImageMagickColorMap["DarkTurquoise"] = "#00CED1"
    ImageMagickColorMap["azure"] = "#F0FFFF"
    ImageMagickColorMap["azure1"] = "#F0FFFF"
    ImageMagickColorMap["LightCyan"] = "#E0FFFF"
    ImageMagickColorMap["LightCyan1"] = "#E0FFFF"
    ImageMagickColorMap["azure2"] = "#E0EEEE"
    ImageMagickColorMap["LightCyan2"] = "#D1EEEE"
    ImageMagickColorMap["PaleTurquoise1"] = "#BBFFFF"
    ImageMagickColorMap["PaleTurquoise"] = "#AFEEEE"
    ImageMagickColorMap["PaleTurquoise2"] = "#AEEEEE"
    ImageMagickColorMap["DarkSlateGray1"] = "#97FFFF"
    ImageMagickColorMap["azure3"] = "#C1CDCD"
    ImageMagickColorMap["LightCyan3"] = "#B4CDCD"
    ImageMagickColorMap["DarkSlateGray2"] = "#8DEEEE"
    ImageMagickColorMap["PaleTurquoise3"] = "#96CDCD"
    ImageMagickColorMap["DarkSlateGray3"] = "#79CDCD"
    ImageMagickColorMap["azure4"] = "#838B8B"
    ImageMagickColorMap["LightCyan4"] = "#7A8B8B"
    ImageMagickColorMap["aqua"] = "#00FFFF"
    ImageMagickColorMap["cyan"] = "#00FFFF"
    ImageMagickColorMap["cyan1"] = "#00FFFF"
    ImageMagickColorMap["PaleTurquoise4"] = "#668B8B"
    ImageMagickColorMap["cyan2"] = "#00EEEE"
    ImageMagickColorMap["DarkSlateGray4"] = "#528B8B"
    ImageMagickColorMap["cyan3"] = "#00CDCD"
    ImageMagickColorMap["cyan4"] = "#008B8B"
    ImageMagickColorMap["DarkCyan"] = "#008B8B"
    ImageMagickColorMap["teal"] = "#008080"
    ImageMagickColorMap["DarkSlateGray"] = "#2F4F4F"
    ImageMagickColorMap["DarkSlateGrey"] = "#2F4F4F"
    ImageMagickColorMap["Turquoise"] = "#48D1CC"
    ImageMagickColorMap["LightSeaGreen"] = "#20B2AA"
    ImageMagickColorMap["turquoise"] = "#40E0D0"
    ImageMagickColorMap["aquamarine4"] = "#458B74"
    ImageMagickColorMap["aquamarine"] = "#7FFFD4"
    ImageMagickColorMap["aquamarine1"] = "#7FFFD4"
    ImageMagickColorMap["aquamarine2"] = "#76EEC6"
    ImageMagickColorMap["aquamarine3"] = "#66CDAA"
    ImageMagickColorMap["Aquamarine"] = "#66CDAA"
    ImageMagickColorMap["SpringGreen"] = "#00FA9A"
    ImageMagickColorMap["MintCream"] = "#F5FFFA"
    ImageMagickColorMap["SpringGreen"] = "#00FF7F"
    ImageMagickColorMap["SpringGreen1"] = "#00FF7F"
    ImageMagickColorMap["SpringGreen2"] = "#00EE76"
    ImageMagickColorMap["SpringGreen3"] = "#00CD66"
    ImageMagickColorMap["SpringGreen4"] = "#008B45"
    ImageMagickColorMap["SeaGreen"] = "#3CB371"
    ImageMagickColorMap["SeaGreen"] = "#2E8B57"
    ImageMagickColorMap["SeaGreen3"] = "#43CD80"
    ImageMagickColorMap["SeaGreen1"] = "#54FF9F"
    ImageMagickColorMap["SeaGreen4"] = "#2E8B57"
    ImageMagickColorMap["SeaGreen2"] = "#4EEE94"
    ImageMagickColorMap["ForestGreen"] = "#32814B"
    ImageMagickColorMap["honeydew"] = "#F0FFF0"
    ImageMagickColorMap["honeydew1"] = "#F0FFF0"
    ImageMagickColorMap["honeydew2"] = "#E0EEE0"
    ImageMagickColorMap["DarkSeaGreen1"] = "#C1FFC1"
    ImageMagickColorMap["DarkSeaGreen2"] = "#B4EEB4"
    ImageMagickColorMap["PaleGreen1"] = "#9AFF9A"
    ImageMagickColorMap["PaleGreen"] = "#98FB98"
    ImageMagickColorMap["honeydew3"] = "#C1CDC1"
    ImageMagickColorMap["LightGreen"] = "#90EE90"
    ImageMagickColorMap["PaleGreen2"] = "#90EE90"
    ImageMagickColorMap["DarkSeaGreen3"] = "#9BCD9B"
    ImageMagickColorMap["DarkSeaGreen"] = "#8FBC8F"
    ImageMagickColorMap["PaleGreen3"] = "#7CCD7C"
    ImageMagickColorMap["honeydew4"] = "#838B83"
    ImageMagickColorMap["green1"] = "#00FF00"
    ImageMagickColorMap["lime"] = "#00FF00"
    ImageMagickColorMap["LimeGreen"] = "#32CD32"
    ImageMagickColorMap["DarkSeaGreen4"] = "#698B69"
    ImageMagickColorMap["green2"] = "#00EE00"
    ImageMagickColorMap["PaleGreen4"] = "#548B54"
    ImageMagickColorMap["green3"] = "#00CD00"
    ImageMagickColorMap["ForestGreen"] = "#228B22"
    ImageMagickColorMap["green4"] = "#008B00"
    ImageMagickColorMap["green"] = "#008000"
    ImageMagickColorMap["DarkGreen"] = "#006400"
    ImageMagickColorMap["LawnGreen"] = "#7CFC00"
    ImageMagickColorMap["chartreuse"] = "#7FFF00"
    ImageMagickColorMap["chartreuse1"] = "#7FFF00"
    ImageMagickColorMap["chartreuse2"] = "#76EE00"
    ImageMagickColorMap["chartreuse3"] = "#66CD00"
    ImageMagickColorMap["chartreuse4"] = "#458B00"
    ImageMagickColorMap["GreenYellow"] = "#ADFF2F"
    ImageMagickColorMap["DarkOliveGreen3"] = "#A2CD5A"
    ImageMagickColorMap["DarkOliveGreen1"] = "#CAFF70"
    ImageMagickColorMap["DarkOliveGreen2"] = "#BCEE68"
    ImageMagickColorMap["DarkOliveGreen4"] = "#6E8B3D"
    ImageMagickColorMap["DarkOliveGreen"] = "#556B2F"
    ImageMagickColorMap["OliveDrab"] = "#6B8E23"
    ImageMagickColorMap["OliveDrab1"] = "#C0FF3E"
    ImageMagickColorMap["OliveDrab2"] = "#B3EE3A"
    ImageMagickColorMap["OliveDrab3"] = "#9ACD32"
    ImageMagickColorMap["YellowGreen"] = "#9ACD32"
    ImageMagickColorMap["OliveDrab4"] = "#698B22"
    ImageMagickColorMap["ivory"] = "#FFFFF0"
    ImageMagickColorMap["ivory1"] = "#FFFFF0"
    ImageMagickColorMap["LightYellow"] = "#FFFFE0"
    ImageMagickColorMap["LightYellow1"] = "#FFFFE0"
    ImageMagickColorMap["beige"] = "#F5F5DC"
    ImageMagickColorMap["ivory2"] = "#EEEEE0"
    ImageMagickColorMap["GoldenYellow"] = "#FAFAD2"
    ImageMagickColorMap["LightYellow2"] = "#EEEED1"
    ImageMagickColorMap["ivory3"] = "#CDCDC1"
    ImageMagickColorMap["LightYellow3"] = "#CDCDB4"
    ImageMagickColorMap["ivory4"] = "#8B8B83"
    ImageMagickColorMap["LightYellow4"] = "#8B8B7A"
    ImageMagickColorMap["yellow"] = "#FFFF00"
    ImageMagickColorMap["yellow1"] = "#FFFF00"
    ImageMagickColorMap["yellow2"] = "#EEEE00"
    ImageMagickColorMap["yellow3"] = "#CDCD00"
    ImageMagickColorMap["yellow4"] = "#8B8B00"
    ImageMagickColorMap["olive"] = "#808000"
    ImageMagickColorMap["DarkKhaki"] = "#BDB76B"
    ImageMagickColorMap["khaki2"] = "#EEE685"
    ImageMagickColorMap["LemonChiffon4"] = "#8B8970"
    ImageMagickColorMap["khaki1"] = "#FFF68F"
    ImageMagickColorMap["khaki3"] = "#CDC673"
    ImageMagickColorMap["khaki4"] = "#8B864E"
    ImageMagickColorMap["PaleGoldenrod"] = "#EEE8AA"
    ImageMagickColorMap["LemonChiffon"] = "#FFFACD"
    ImageMagickColorMap["LemonChiffon1"] = "#FFFACD"
    ImageMagickColorMap["khaki"] = "#F0E68C"
    ImageMagickColorMap["LemonChiffon3"] = "#CDC9A5"
    ImageMagickColorMap["LemonChiffon2"] = "#EEE9BF"
    ImageMagickColorMap["GoldenRod"] = "#D1C166"
    ImageMagickColorMap["cornsilk4"] = "#8B8878"
    ImageMagickColorMap["gold"] = "#FFD700"
    ImageMagickColorMap["gold1"] = "#FFD700"
    ImageMagickColorMap["gold2"] = "#EEC900"
    ImageMagickColorMap["gold3"] = "#CDAD00"
    ImageMagickColorMap["gold4"] = "#8B7500"
    ImageMagickColorMap["LightGoldenrod"] = "#EEDD82"
    ImageMagickColorMap["LightGoldenrod4"] = "#8B814C"
    ImageMagickColorMap["LightGoldenrod1"] = "#FFEC8B"
    ImageMagickColorMap["LightGoldenrod3"] = "#CDBE70"
    ImageMagickColorMap["LightGoldenrod2"] = "#EEDC82"
    ImageMagickColorMap["cornsilk3"] = "#CDC8B1"
    ImageMagickColorMap["cornsilk2"] = "#EEE8CD"
    ImageMagickColorMap["cornsilk"] = "#FFF8DC"
    ImageMagickColorMap["cornsilk1"] = "#FFF8DC"
    ImageMagickColorMap["goldenrod"] = "#DAA520"
    ImageMagickColorMap["goldenrod1"] = "#FFC125"
    ImageMagickColorMap["goldenrod2"] = "#EEB422"
    ImageMagickColorMap["goldenrod3"] = "#CD9B1D"
    ImageMagickColorMap["goldenrod4"] = "#8B6914"
    ImageMagickColorMap["DarkGoldenrod"] = "#B8860B"
    ImageMagickColorMap["DarkGoldenrod1"] = "#FFB90F"
    ImageMagickColorMap["DarkGoldenrod2"] = "#EEAD0E"
    ImageMagickColorMap["DarkGoldenrod3"] = "#CD950C"
    ImageMagickColorMap["DarkGoldenrod4"] = "#8B6508"
    ImageMagickColorMap["FloralWhite"] = "#FFFAF0"
    ImageMagickColorMap["wheat2"] = "#EED8AE"
    ImageMagickColorMap["OldLace"] = "#FDF5E6"
    ImageMagickColorMap["wheat"] = "#F5DEB3"
    ImageMagickColorMap["wheat1"] = "#FFE7BA"
    ImageMagickColorMap["wheat3"] = "#CDBA96"
    ImageMagickColorMap["orange"] = "#FFA500"
    ImageMagickColorMap["orange1"] = "#FFA500"
    ImageMagickColorMap["orange2"] = "#EE9A00"
    ImageMagickColorMap["orange3"] = "#CD8500"
    ImageMagickColorMap["orange4"] = "#8B5A00"
    ImageMagickColorMap["wheat4"] = "#8B7E66"
    ImageMagickColorMap["moccasin"] = "#FFE4B5"
    ImageMagickColorMap["PapayaWhip"] = "#FFEFD5"
    ImageMagickColorMap["Sandy"] = "#CDB38B"
    ImageMagickColorMap["BlanchedAlmond"] = "#FFEBCD"
    ImageMagickColorMap["Sandy"] = "#FFDEAD"
    ImageMagickColorMap["Sandy1"] = "#FFDEAD"
    ImageMagickColorMap["Sandy2"] = "#EECFA1"
    ImageMagickColorMap["Sandy4"] = "#8B795E"
    ImageMagickColorMap["AntiqueWhite4"] = "#8B8378"
    ImageMagickColorMap["AntiqueWhite"] = "#FAEBD7"
    ImageMagickColorMap["tan"] = "#D2B48C"
    ImageMagickColorMap["bisque4"] = "#8B7D6B"
    ImageMagickColorMap["burlywood"] = "#DEB887"
    ImageMagickColorMap["AntiqueWhite2"] = "#EEDFCC"
    ImageMagickColorMap["burlywood1"] = "#FFD39B"
    ImageMagickColorMap["burlywood3"] = "#CDAA7D"
    ImageMagickColorMap["burlywood2"] = "#EEC591"
    ImageMagickColorMap["AntiqueWhite1"] = "#FFEFDB"
    ImageMagickColorMap["burlywood4"] = "#8B7355"
    ImageMagickColorMap["AntiqueWhite3"] = "#CDC0B0"
    ImageMagickColorMap["DarkOrange"] = "#FF8C00"
    ImageMagickColorMap["bisque2"] = "#EED5B7"
    ImageMagickColorMap["bisque"] = "#FFE4C4"
    ImageMagickColorMap["bisque1"] = "#FFE4C4"
    ImageMagickColorMap["bisque3"] = "#CDB79E"
    ImageMagickColorMap["DarkOrange1"] = "#FF7F00"
    ImageMagickColorMap["linen"] = "#FAF0E6"
    ImageMagickColorMap["DarkOrange2"] = "#EE7600"
    ImageMagickColorMap["DarkOrange3"] = "#CD6600"
    ImageMagickColorMap["DarkOrange4"] = "#8B4500"
    ImageMagickColorMap["peru"] = "#CD853F"
    ImageMagickColorMap["tan1"] = "#FFA54F"
    ImageMagickColorMap["tan2"] = "#EE9A49"
    ImageMagickColorMap["tan3"] = "#CD853F"
    ImageMagickColorMap["tan4"] = "#8B5A2B"
    ImageMagickColorMap["PeachPuff"] = "#FFDAB9"
    ImageMagickColorMap["PeachPuff1"] = "#FFDAB9"
    ImageMagickColorMap["PeachPuff4"] = "#8B7765"
    ImageMagickColorMap["PeachPuff2"] = "#EECBAD"
    ImageMagickColorMap["PeachPuff3"] = "#CDAF95"
    ImageMagickColorMap["SandyBrown"] = "#F4A460"
    ImageMagickColorMap["seashell4"] = "#8B8682"
    ImageMagickColorMap["seashell2"] = "#EEE5DE"
    ImageMagickColorMap["seashell3"] = "#CDC5BF"
    ImageMagickColorMap["chocolate"] = "#D2691E"
    ImageMagickColorMap["chocolate1"] = "#FF7F24"
    ImageMagickColorMap["chocolate2"] = "#EE7621"
    ImageMagickColorMap["chocolate3"] = "#CD661D"
    ImageMagickColorMap["chocolate4"] = "#8B4513"
    ImageMagickColorMap["SaddleBrown"] = "#8B4513"
    ImageMagickColorMap["seashell"] = "#FFF5EE"
    ImageMagickColorMap["seashell1"] = "#FFF5EE"
    ImageMagickColorMap["sienna4"] = "#8B4726"
    ImageMagickColorMap["sienna"] = "#A0522D"
    ImageMagickColorMap["sienna1"] = "#FF8247"
    ImageMagickColorMap["sienna2"] = "#EE7942"
    ImageMagickColorMap["sienna3"] = "#CD6839"
    ImageMagickColorMap["LightSalmon3"] = "#CD8162"
    ImageMagickColorMap["LightSalmon"] = "#FFA07A"
    ImageMagickColorMap["LightSalmon1"] = "#FFA07A"
    ImageMagickColorMap["LightSalmon4"] = "#8B5742"
    ImageMagickColorMap["LightSalmon2"] = "#EE9572"
    ImageMagickColorMap["coral"] = "#FF7F50"
    ImageMagickColorMap["OrangeRed"] = "#FF4500"
    ImageMagickColorMap["OrangeRed1"] = "#FF4500"
    ImageMagickColorMap["OrangeRed2"] = "#EE4000"
    ImageMagickColorMap["OrangeRed3"] = "#CD3700"
    ImageMagickColorMap["OrangeRed4"] = "#8B2500"
    ImageMagickColorMap["DarkSalmon"] = "#E9967A"
    ImageMagickColorMap["salmon1"] = "#FF8C69"
    ImageMagickColorMap["salmon2"] = "#EE8262"
    ImageMagickColorMap["salmon3"] = "#CD7054"
    ImageMagickColorMap["salmon4"] = "#8B4C39"
    ImageMagickColorMap["coral1"] = "#FF7256"
    ImageMagickColorMap["coral2"] = "#EE6A50"
    ImageMagickColorMap["coral3"] = "#CD5B45"
    ImageMagickColorMap["coral4"] = "#8B3E2F"
    ImageMagickColorMap["tomato4"] = "#8B3626"
    ImageMagickColorMap["tomato"] = "#FF6347"
    ImageMagickColorMap["tomato1"] = "#FF6347"
    ImageMagickColorMap["tomato2"] = "#EE5C42"
    ImageMagickColorMap["tomato3"] = "#CD4F39"
    ImageMagickColorMap["MistyRose4"] = "#8B7D7B"
    ImageMagickColorMap["MistyRose2"] = "#EED5D2"
    ImageMagickColorMap["MistyRose"] = "#FFE4E1"
    ImageMagickColorMap["MistyRose1"] = "#FFE4E1"
    ImageMagickColorMap["salmon"] = "#FA8072"
    ImageMagickColorMap["MistyRose3"] = "#CDB7B5"
    ImageMagickColorMap["white"] = "#FFFFFF"
    ImageMagickColorMap["gray100"] = "#FFFFFF"
    ImageMagickColorMap["grey100"] = "#FFFFFF"
    ImageMagickColorMap["grey100"] = "#FFFFFF"
    ImageMagickColorMap["gray99"] = "#FCFCFC"
    ImageMagickColorMap["grey99"] = "#FCFCFC"
    ImageMagickColorMap["WhiteSmoke"] = "#F5F5F5"
    ImageMagickColorMap["LightPink5"] = "#EDEDED"
    ImageMagickColorMap["LightPink6"] = "#EBEBEB"
    ImageMagickColorMap["gray91"] = "#E8E8E8"
    ImageMagickColorMap["grey91"] = "#E8E8E8"
    ImageMagickColorMap["gray90"] = "#E5E5E5"
    ImageMagickColorMap["grey90"] = "#E5E5E5"
    ImageMagickColorMap["gray89"] = "#E3E3E3"
    ImageMagickColorMap["grey89"] = "#E3E3E3"
    ImageMagickColorMap["gray88"] = "#E0E0E0"
    ImageMagickColorMap["grey88"] = "#E0E0E0"
    ImageMagickColorMap["gray87"] = "#DEDEDE"
    ImageMagickColorMap["grey87"] = "#DEDEDE"
    ImageMagickColorMap["gainsboro"] = "#DCDCDC"
    ImageMagickColorMap["gray86"] = "#DBDBDB"
    ImageMagickColorMap["grey86"] = "#DBDBDB"
    ImageMagickColorMap["gray85"] = "#D9D9D9"
    ImageMagickColorMap["grey85"] = "#D9D9D9"
    ImageMagickColorMap["gray84"] = "#D6D6D6"
    ImageMagickColorMap["grey84"] = "#D6D6D6"
    ImageMagickColorMap["gray83"] = "#D4D4D4"
    ImageMagickColorMap["grey83"] = "#D4D4D4"
    ImageMagickColorMap["LightGray"] = "#D3D3D3"
    ImageMagickColorMap["LightGrey"] = "#D3D3D3"
    ImageMagickColorMap["gray82"] = "#D1D1D1"
    ImageMagickColorMap["grey82"] = "#D1D1D1"
    ImageMagickColorMap["gray81"] = "#CFCFCF"
    ImageMagickColorMap["grey81"] = "#CFCFCF"
    ImageMagickColorMap["gray80"] = "#CCCCCC"
    ImageMagickColorMap["grey80"] = "#CCCCCC"
    ImageMagickColorMap["gray79"] = "#C9C9C9"
    ImageMagickColorMap["grey79"] = "#C9C9C9"
    ImageMagickColorMap["gray78"] = "#C7C7C7"
    ImageMagickColorMap["grey78"] = "#C7C7C7"
    ImageMagickColorMap["gray77"] = "#C4C4C4"
    ImageMagickColorMap["grey77"] = "#C4C4C4"
    ImageMagickColorMap["gray76"] = "#C2C2C2"
    ImageMagickColorMap["grey76"] = "#C2C2C2"
    ImageMagickColorMap["silver"] = "#C0C0C0"
    ImageMagickColorMap["gray75"] = "#BFBFBF"
    ImageMagickColorMap["grey75"] = "#BFBFBF"
    ImageMagickColorMap["gray74"] = "#BDBDBD"
    ImageMagickColorMap["grey74"] = "#BDBDBD"
    ImageMagickColorMap["gray73"] = "#BABABA"
    ImageMagickColorMap["grey73"] = "#BABABA"
    ImageMagickColorMap["gray72"] = "#B8B8B8"
    ImageMagickColorMap["grey72"] = "#B8B8B8"
    ImageMagickColorMap["gray71"] = "#B5B5B5"
    ImageMagickColorMap["grey71"] = "#B5B5B5"
    ImageMagickColorMap["gray70"] = "#B3B3B3"
    ImageMagickColorMap["grey70"] = "#B3B3B3"
    ImageMagickColorMap["gray69"] = "#B0B0B0"
    ImageMagickColorMap["grey69"] = "#B0B0B0"
    ImageMagickColorMap["gray68"] = "#ADADAD"
    ImageMagickColorMap["grey68"] = "#ADADAD"
    ImageMagickColorMap["gray67"] = "#ABABAB"
    ImageMagickColorMap["grey67"] = "#ABABAB"
    ImageMagickColorMap["DarkGray"] = "#A9A9A9"
    ImageMagickColorMap["DarkGrey"] = "#A9A9A9"
    ImageMagickColorMap["gray66"] = "#A8A8A8"
    ImageMagickColorMap["grey66"] = "#A8A8A8"
    ImageMagickColorMap["gray65"] = "#A6A6A6"
    ImageMagickColorMap["grey65"] = "#A6A6A6"
    ImageMagickColorMap["gray64"] = "#A3A3A3"
    ImageMagickColorMap["grey64"] = "#A3A3A3"
    ImageMagickColorMap["gray63"] = "#A1A1A1"
    ImageMagickColorMap["grey63"] = "#A1A1A1"
    ImageMagickColorMap["gray62"] = "#9E9E9E"
    ImageMagickColorMap["grey62"] = "#9E9E9E"
    ImageMagickColorMap["gray61"] = "#9C9C9C"
    ImageMagickColorMap["grey61"] = "#9C9C9C"
    ImageMagickColorMap["gray60"] = "#999999"
    ImageMagickColorMap["grey60"] = "#999999"
    ImageMagickColorMap["gray59"] = "#969696"
    ImageMagickColorMap["grey59"] = "#969696"
    ImageMagickColorMap["gray58"] = "#949494"
    ImageMagickColorMap["grey58"] = "#949494"
    ImageMagickColorMap["gray57"] = "#919191"
    ImageMagickColorMap["grey57"] = "#919191"
    ImageMagickColorMap["gray56"] = "#8F8F8F"
    ImageMagickColorMap["grey56"] = "#8F8F8F"
    ImageMagickColorMap["gray55"] = "#8C8C8C"
    ImageMagickColorMap["grey55"] = "#8C8C8C"
    ImageMagickColorMap["gray54"] = "#8A8A8A"
    ImageMagickColorMap["grey54"] = "#8A8A8A"
    ImageMagickColorMap["gray53"] = "#878787"
    ImageMagickColorMap["grey53"] = "#878787"
    ImageMagickColorMap["gray52"] = "#858585"
    ImageMagickColorMap["grey52"] = "#858585"
    ImageMagickColorMap["gray51"] = "#828282"
    ImageMagickColorMap["grey51"] = "#828282"
    ImageMagickColorMap["fractal"] = "#808080"
    ImageMagickColorMap["gray50"] = "#7F7F7F"
    ImageMagickColorMap["grey50"] = "#7F7F7F"
    ImageMagickColorMap["gray"] = "#7E7E7E"
    ImageMagickColorMap["gray49"] = "#7D7D7D"
    ImageMagickColorMap["grey49"] = "#7D7D7D"
    ImageMagickColorMap["gray48"] = "#7A7A7A"
    ImageMagickColorMap["grey48"] = "#7A7A7A"
    ImageMagickColorMap["gray47"] = "#787878"
    ImageMagickColorMap["grey47"] = "#787878"
    ImageMagickColorMap["gray46"] = "#757575"
    ImageMagickColorMap["grey46"] = "#757575"
    ImageMagickColorMap["gray45"] = "#737373"
    ImageMagickColorMap["grey45"] = "#737373"
    ImageMagickColorMap["gray44"] = "#707070"
    ImageMagickColorMap["grey44"] = "#707070"
    ImageMagickColorMap["gray43"] = "#6E6E6E"
    ImageMagickColorMap["grey43"] = "#6E6E6E"
    ImageMagickColorMap["gray42"] = "#6B6B6B"
    ImageMagickColorMap["grey42"] = "#6B6B6B"
    ImageMagickColorMap["DimGray"] = "#696969"
    ImageMagickColorMap["DimGrey"] = "#696969"
    ImageMagickColorMap["gray41"] = "#696969"
    ImageMagickColorMap["grey41"] = "#696969"
    ImageMagickColorMap["gray40"] = "#666666"
    ImageMagickColorMap["grey40"] = "#666666"
    ImageMagickColorMap["gray39"] = "#636363"
    ImageMagickColorMap["grey39"] = "#636363"
    ImageMagickColorMap["gray38"] = "#616161"
    ImageMagickColorMap["grey38"] = "#616161"
    ImageMagickColorMap["gray37"] = "#5E5E5E"
    ImageMagickColorMap["grey37"] = "#5E5E5E"
    ImageMagickColorMap["gray36"] = "#5C5C5C"
    ImageMagickColorMap["grey36"] = "#5C5C5C"
    ImageMagickColorMap["gray35"] = "#595959"
    ImageMagickColorMap["grey35"] = "#595959"
    ImageMagickColorMap["gray34"] = "#575757"
    ImageMagickColorMap["gray33"] = "#545454"
    ImageMagickColorMap["grey32"] = "#525252"
    ImageMagickColorMap["grey31"] = "#4F4F4F"
    ImageMagickColorMap["grey30"] = "#4D4D4D"
    ImageMagickColorMap["grey29"] = "#4A4A4A"
    ImageMagickColorMap["gray28"] = "#474747"
    ImageMagickColorMap["gray27"] = "#454545"
    ImageMagickColorMap["gray26"] = "#424242"
    ImageMagickColorMap["gray25"] = "#404040"
    ImageMagickColorMap["gray24"] = "#3D3D3D"
    ImageMagickColorMap["gray23"] = "#3B3B3B"
    ImageMagickColorMap["gray22"] = "#383838"
    ImageMagickColorMap["gray21"] = "#363636"
    ImageMagickColorMap["black"] = "#130f17"
    ImageMagickColorMap["black"] = "#333333"
    ImageMagickColorMap["black"] = "#303030"
    ImageMagickColorMap["black"] = "#2E2E2E"
    ImageMagickColorMap["black"] = "#2B2B2B"
    ImageMagickColorMap["black"] = "#292929"
    ImageMagickColorMap["black"] = "#262626"
    ImageMagickColorMap["black"] = "#242424"
    ImageMagickColorMap["black"] = "#212121"
    ImageMagickColorMap["black"] = "#1F1F1F"
    ImageMagickColorMap["black"] = "#1C1C1C"
    ImageMagickColorMap["black"] = "#1A1A1A"
    ImageMagickColorMap["black"] = "#171717"
    ImageMagickColorMap["black"] = "#141414"
    ImageMagickColorMap["black"] = "#121212"
    ImageMagickColorMap["black"] = "#000000"
    ImageMagickColorMap["black"] = "#0d0d15"

    all_the_colors = {**WebColorMap, **ImageMagickColorMap}
    
    @staticmethod
    def rgbFromStr(s):  
        # s starts with a #.  
        r, g, b = int(s[1:3],16), int(s[3:5], 16),int(s[5:7], 16)  
        # print("r, g, b", r, g, b)
        return r, g, b  
    
    @staticmethod
    def findNearestWebColorName(RGB_tuple):  
        return ColorNames.findNearestColorName((R,G,B),ColorNames.all_the_colors)
    
    @staticmethod
    def findNearestImageMagickColorName(RGB_tuple):  
        return ColorNames.findNearestColorName(RGB_tuple,ColorNames.ImageMagickColorMap)
    
    @staticmethod
    def findNearestColorName(RGB_tuple,Map):  
        mindiff = None
        for d in Map:  
            r, g, b = ColorNames.rgbFromStr(Map[d])  
            diff = abs(RGB_tuple[0] -r)*256 + abs(RGB_tuple[1]-g)* 256 + abs(RGB_tuple[2]- b)* 256   
            if mindiff is None or diff < mindiff:  
                mindiff = diff  
                mincolorname = d  
        return mincolorname