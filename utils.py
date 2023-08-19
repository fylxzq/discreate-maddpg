import xlrd
import xlwt
import os
from xlutils import copy


def writeToExcel(datas,filename,colname):

    book = xlwt.Workbook(encoding="utf-8",style_compression=0)
    sheet = book.add_sheet("sheet1",cell_overwrite_ok=True)
    data_len = len(datas)
    for i in range(0,data_len):
        data = datas[i]
        for j in range(0,len(colname)):
            sheet.write(i,j,data[j])
    book.save(filename)

def writeRedundancy(datas,step,filename):
    if(not os.path.exists(filename)):
        writeToExcel([],filename,[])
    xls = xlrd.open_workbook(filename)
    wbook = copy.copy(xls)
    wsheet = wbook.get_sheet(0)
    for i in range(len(datas)):
        wsheet.write(i,step,datas[i])
    wbook.save(filename)



def readFromExcel(filename):
    data = xlrd.open_workbook(filename)
    sheet = data.sheets()[0]
    n_cols = sheet.ncols
    result = []
    for i in range(0,n_cols):
        result.append(sheet.col_values(i))
    return result

def getTargetInfo(env):
    hit_nums = 0
    request_nums = 0
    transmission_size = 0
    for base in env.basestations:
        hit_nums += base.hit_num
        request_nums += base.request_num
        transmission_size += base.transmission_size
        base.hit_num = 0
        base.request_num = 0
        base.transmission_size = 0
    return [hit_nums/request_nums,transmission_size]

# if __name__ == '__main__':

    #writeRedundancy([2,3,0.1,0.2],1,"datas/redundancy_datas/1.xls")