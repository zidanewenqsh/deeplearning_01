'''
@Descripttion: 百度图片爬虫
@version: v0.1
@Author: QsWen
@Date: 2020-04-28 16:35:04
@LastEditors: QsWen
@LastEditTime: 2020-04-28 16:41:12
'''
import urllib.request
import urllib.parse
import re
import os
# 爬虫
# 保存目录
img_savedir = r"D:\PycharmProjects\deeplearning_01\yolo_v4\datas\catdog"
# 分类
cls = 0
# 图片数
img_num = 3000
#添加header，其中Referer是必须的,否则会返回403错误，User-Agent是必须的，这样才可以伪装成浏览器进行访问
header=\
{
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
     "referer":"https://image.baidu.com"
    }
url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&word={word}&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&cg=girl&pn={pageNum}&rn=30&gsm=1e00000000001e&1490169411926="
keyword = input("请输入搜索关键字：")
#转码
keyword = urllib.parse.quote(keyword,'utf-8')

n = 0
j = 0

while(n<img_num):
    error = 0
    n+=30
    #url
    url1 = url.format(word=keyword,pageNum=str(n))
    #获取请求
    rep = urllib.request.Request(url1,headers=header)
    #打开网页
    rep = urllib.request.urlopen(rep)
    #获取网页内容
    try:
        html = rep.read().decode('utf-8')
        # print(html)
    except:
        print("出错了！")
        error = 1
        print("出错页数："+str(n))
    if error == 1:
        continue
    #正则匹配
    p = re.compile("thumbURL.*?\.jpg")
    #获取正则匹配到的结果，返回list
    s = p.findall(html)
    if os.path.isdir("D://pic") != True:
        os.makedirs("D://pic")
    with open("testpic.txt","a") as f:
        #获取图片
        for i in s:
            print(i)
            i = i.replace('thumbURL":"','')
            print(i)
            f.write(i)
            f.write("\n")
            #保存图片
            urllib.request.urlretrieve(i,os.path.join(img_savedir,"{cls}_{num}.jpg".format(cls=cls,num=j)))
            j+=1
        f.close()
print("总共爬取图片数为："+str(j))