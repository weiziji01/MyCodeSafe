from osgeo import gdal
import os
 
 
def TIFToPNG(tifDir_path, pngDir_path):
    for fileName in os.listdir(tifDir_path):
        if fileName[-4:] == ".tif":
            ds = gdal.Open(tifDir_path + fileName)
            driver = gdal.GetDriverByName('PNG')
            driver.CreateCopy(pngDir_path + fileName[:-4] + ".png", ds)
            print("已生成：",pngDir_path + fileName[:-4] + ".png")
 
if __name__ == '__main__':
    tifDir_path = "/qiyuan_data/"
    pngDir_path = "/qiyuan_data/png/"
    TIFToPNG(tifDir_path, pngDir_path)