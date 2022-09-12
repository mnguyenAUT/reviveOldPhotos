import os, sys, cv2
import numpy as np
from wand.image import Image
from PIL import Image as PilI
images = []

def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

if len(sys.argv) != 2:
    print("Usage:")
    print("python revivePhotos.py [input image]")
    print("For example: 'python revivePhotos.py download.jpg'")
    exit(0)

imgQR = cv2.imread(sys.argv[1])
b, g, r = cv2.split(imgQR)

cv2.imwrite("r.png", r)
cv2.imwrite("g.png", g)
cv2.imwrite("b.png", b)

os.chdir("./colorization")

os.system("python demo_release.py -i ../r.png")
one = cv2.imread("saved_eccv16.png")
two = cv2.imread("saved_siggraph17.png")
leftImage = cv2.addWeighted(one,0.5,two,0.5,0)
cv2.imwrite("../r_C.png", leftImage)

os.system("python demo_release.py -i ../g.png")
one = cv2.imread("saved_eccv16.png")
two = cv2.imread("saved_siggraph17.png")
leftImage = cv2.addWeighted(one,0.5,two,0.5,0)
cv2.imwrite("../g_C.png", leftImage)

os.system("python demo_release.py -i ../b.png")
one = cv2.imread("saved_eccv16.png")
two = cv2.imread("saved_siggraph17.png")
leftImage = cv2.addWeighted(one,0.5,two,0.5,0)
cv2.imwrite("../b_C.png", leftImage)

#rgb = cv2.addWeighted(one,0.5,two,0.5,0)

os.chdir("../")

os.chdir("./GFPGAN")
os.system("python inference_gfpgan.py -i ../r_C.png -v 1.3")
os.system("python inference_gfpgan.py -i ../g_C.png -v 1.3")
os.system("python inference_gfpgan.py -i ../b_C.png -v 1.3")

r_c_image = cv2.imread("./results/restored_imgs/r_C.png")
g_c_image = cv2.imread("./results/restored_imgs/g_C.png")
b_c_image = cv2.imread("./results/restored_imgs/b_C.png")
finalImage = cv2.addWeighted(r_c_image,0.3333,cv2.addWeighted(g_c_image,0.5,b_c_image,0.5,0),0.6666,0)
finalImage = white_balance_loops(finalImage)
cv2.imwrite("../finalImage.jpg", finalImage)



