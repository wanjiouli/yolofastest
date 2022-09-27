import cv2
import numpy as np
# 设置窗口宽大小
win_width = 1920
# 设置窗口高大小
win_height = 1200
#
mid_width = int(win_width / 2)
mid_height = int(win_height / 2)

# 设置焦距大小
foc = 2810.0
# A4的宽长
real_wid = 11.69
# 字体
font = cv2.FONT_HERSHEY_SIMPLEX # 0
print('***********************')
print(font,type(font))
print('***********************')
w_ok = 1

# 打开摄像头
capture = cv2.VideoCapture(0)
capture.set(3, win_width)
capture.set(4, win_height)

while (True):
    # 获取每一帧
    ret, frame = capture.read()
    # frame = cv2.flip(frame, 1)

    if ret == False:
        break
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯滤波用的是5x5的卷积核 在x轴上的卷积核的标准差是0
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 图像阈值处理 第一个参数是图片,第二个参数是阈值,第三个参数表示最大值,第四个参数是划分使用的是什么类型的算法 常用值为0（cv2.THRESH_BINARY）
    ret, binary = cv2.threshold(gray, 127, 255, 0)

    # 返回指定形状和尺寸的结构元素
    # 这个函数的第一个参数表示内核的形状，有三种形状可以选择。
    # 矩形：MORPH_RECT;
    # 交叉形：MORPH_CROSS;
    # 椭圆形：MORPH_ELLIPSE;
    # 第二和第三个参数分别是内核的尺寸以及锚点的位置。 对于锚点的位置，有默认值Point（-1,-1）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 膨胀处理 第一个参数是目标图片,第二个参数是机械能操作的内核,默认是3x3的矩阵,第三个参数是腐蚀次数
    # 将前景物体变大，理解成将图像断开裂缝变小（在图片上画上黑色印记，印记越来越小）
    binary = cv2.dilate(binary, kernel, iterations=2)  # 形态学膨胀

    # 图像轮廓检测 第一个参数是寻找轮廓的图像 第二个参数是轮廓的检索模式 第三个参数是为轮廓的近似办法
    # 检索模式有四种模式,如下:
    # cv2.RETR_EXTERNAL 表示只检测外轮廓
    # cv2.RETR_LIST     检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE     建立一个等级树结构的轮廓。
    # 轮廓的近似办法有三,如下:
    # cv2.CHAIN_APPROX_NONE       存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE     压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1    CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinlchain近似算法
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(contours))
    # print(contours[0])
    # cv2.findContours返回两个返回值contours, hierarchy
    # 函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示
    # 该函数还可返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，
    # 每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
    # 分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。


    # 循环取出矩形轮廓信息
    for c in contours:
        # 输入四个点计算矩形面积
        if cv2.contourArea(c) < 2000:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
            continue

        # 得到矩形的左上坐标点与宽高
        x, y, w, h = cv2.boundingRect(c)  # 该函数计算矩形的边界框

        # 如果矩形的坐标点x与y任意一个大于了窗口宽高的一半，则进行下一次的循环
        if x > mid_width or y > mid_height:
            continue

        # 如果坐标点x与y各加上自己的矩形宽高的值任意一个小于窗口的一半，则进行下一次的循环
        if (x + w) < mid_width or (y + h) < mid_height:
            continue

        # 如果矩形的高大与矩形的宽,则进行下一次的循环
        if h > w:
            continue

        # 如果矩形的坐标点x与y任意一个等于0，则进行下一次循环
        if x == 0 or y == 0:
            continue

        # 如果矩形的坐标点x与y任意一个等于窗口的宽高，则进行下一次循环
        if x == win_width or y == win_height:
            continue

        # 将矩形的宽赋值给w_ok
        w_ok = w

        # 画出矩形 第一个参数是原图,第二个参数是图像中的矩形左上的坐标点,第三个参数矩形的宽高,第四个参数是对应的rbg的颜色参数,第五个是字体线条的大小
        cv2.rectangle(frame, (x + 1, y + 1), (x + w_ok - 1, y + h - 1), (0, 255, 0), 2)

    # W是A4纸真实宽 F是焦距 P是像素宽度
    # 计算出来A4距相机的距离英寸   D‘ = (W * F) / P
    dis_inch = (real_wid * foc) / (w_ok - 2)

    # 将英寸转化为cm 就是将 inch * 2.54 = cm
    dis_cm = dis_inch * 2.54

    # 加文字函数 第一个参数是图片,第二个参数是要添加的文字,第三个参数文字要添加到图上的位置坐标,第四个是字体的类型,第五个是字体大小,第六个字体颜色,第七个字体粗细
    frame = cv2.putText(frame, "%.2fcm" % (dis_cm), (5, 25), font, 0.8, (0, 255, 0), 2)
    frame = cv2.putText(frame, "+", (mid_width, mid_height), font, 1.0, (0, 255, 0), 2)

    # 设置窗口名称
    cv2.namedWindow('res', 0)
    cv2.namedWindow('gray', 0)

    # 设置窗口的大小
    cv2.resizeWindow('res', win_width, win_height)
    cv2.resizeWindow('gray', win_width, win_height)

    # 展示窗口
    cv2.imshow('res', frame)
    cv2.imshow('gray', binary)

    # 键盘绑定函数,等待输入值
    c = cv2.waitKey(40)

    # 如果输入值是27,则跳出大循环
    if c == 27:
        break

# 删除所有窗口
cv2.destroyAllWindows()


