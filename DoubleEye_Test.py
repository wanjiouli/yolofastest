import sys
import cv2
import numpy as np
import stereoconfig
import pcl
# import pcl.pcl_visualization
import open3d as o3d


# 预处理
def preprocess(img1, img2):

    # 彩色图->灰度图
    if (img1.ndim == 3):
        # cv2.imread()接口读图像，读进来直接是BGR格式数据格式在0-255，不是常见的RGB格式
        # cv2.cvtColor()函数是颜色空间转换函数
        # 第一个参数是传入需要转化的图片
        # 第二个参数是传入转化为何种格式
        # 有三种格式,如下:
        # cv2.COLOR_BGR2RGB
        # cv2.COLOR_BGR2GRAY
        # cv2.COLOR_BGR2HSV
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    # 图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    # 双目相机的立体校正
    # 第一个参数传入左摄像机相机内参矩阵,矩阵第三行格式应该为0 0 1
    # [[fx,0,fx],
    #  [0,fy,fy],
    #  [0,0,1]]
    # 第二个参数传入左摄像机的畸变向量
    # 第三个参数传入右摄像机相机内参矩阵,矩阵第三行格式应该为0 0 1
    # 第四个参数传入左摄像机的畸变向量
    # 第五个参数传入图像大小
    # 第六个参数传入相机之间的旋转矩阵 这个矩阵的意义计算相机1可通过变换R达到相机2的位置
    # 第七个参数传入平移矩阵
    # alpha参数是拉伸参数。设置为小于0，则不进行拉伸。为0那么校正后图像只有有效部分会被显示。如果设置为1，那么就会显示整个图像。
    # 设置为0-1之间的某个值，其效果也居于两者之间。
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=0)

    # 函数计算无畸变和修正转换映射
    # 第一个参数是传入相机内参矩阵
    # 第二个参数是传入相机的畸变系数 如果这个向量是空的，就认为是零畸变系数。
    # 第三个参数是可选的修正变换参数，传入是个3x3的矩阵，通过stereoRectify计算得来的R1或R2可以放在这里。如果这个矩阵是空的，就假设是单位矩阵。
    # 第四个参数是传入新相机的矩阵P1
    # 第五个参数是传入为畸变的图像尺寸
    # 第六个参数是传入第一个输出的映射的类型，可以为 CV_32FC1, CV_32FC2或CV_16SC2
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    # 重映射函数 其意义是重映射是从图像中的一个位置获取像素并将其放置在新图像中的另一位置的过程。
    # 第一个参数传入输入图像
    # 第二个参数传入(x,y)的一个映射，其值为x
    # 第三个参数传入(x,y)的一个映射，其值为y
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        # cv2.line()函数用于在图像中划线
        # 第一个参数传入要划的线所在图像
        # 第二个参数传入直线起点
        # 第三个参数传入直线终点
        # 第四个参数传入直线的颜色
        # 第五个参数传入线条的粗细
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 15,
              'speckleWindowSize': 100,
              'speckleRange': 1,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 将h×w×3数组转换为N×3的数组
def hw3ToN3(points):
    height, width = points.shape[0:2]

    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)

    points_ = np.hstack((points_1, points_2, points_3))

    return points_


# 深度、颜色转换为点云
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols

    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)

    # 颜色信息
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)

    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)

    # 将坐标+颜色叠加为点云数组
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)

    # 删掉一些不合适的点
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]

    # 下面参数是经验性取值，需要根据实际情况调整
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack(
        (remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))

    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)

    return pointcloud_1


# 点云显示
# def view_cloud(pointcloud):
#     cloud = pcl.PointCloud_PointXYZRGBA()
#     cloud.from_array(pointcloud)
#
#     try:
#         visual = pcl.pcl_visualization.CloudViewing()
#         visual.ShowColorACloud(cloud)
#         v = True
#         while v:
#             v = not (visual.WasStopped())
#     except:
#         pass


def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoconfig.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    # fb 除以 disparityMap + doffs数组，这里会使得fb广播为disparityMap + doffs一样的维度进行除法
    depthMap = np.divide(fb, disparityMap + doffs)
    # np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
    # np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
    # np.logical_or逻辑或
    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)


if __name__ == '__main__':
    # 读取MiddleBurry数据集的图片
    # iml = cv2.imread('Adirondack-perfect/im0.png', 1)  # 左图
    # imr = cv2.imread('Adirondack-perfect/im1.png', 1)  # 右图
    iml = cv2.imread('./im0.png', 1)  # 左图
    imr = cv2.imread('./im1.png', 1)  # 右图
    if (iml is None) or (imr is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    # 读取图片的高宽
    height, width = iml.shape[0:2]


    # 读取相机内参和外参
    # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
    # 初始化类
    config = stereoconfig.stereoCamera()
    # 调用方法
    config.setMiddleBurryParams()
    print(config.cam_matrix_left)

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    print(Q)

    # 绘制等间距平行线，检查立体校正的效果
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('./check_rectification.png', line)

    # 立体匹配
    iml_, imr_ = preprocess(iml, imr)  # 预处理，一般可以削弱光照不均的影响，不做也可以
    disp, _ = stereoMatchSGBM(iml, imr, False)  # 这里传入的是未经立体校正的图像，因为我们使用的middleburry图片已经是校正过的了
    cv2.imwrite('./disaprity.png', disp * 4)

    # 计算深度图
    # depthMap = getDepthMapWithQ(disp, Q)
    depthMap = getDepthMapWithConfig(disp, config)
    minDepth = np.min(depthMap)
    maxDepth = np.max(depthMap)
    print(minDepth, maxDepth)
    depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)
    cv2.imshow("DepthMap", depthMapVis)
    cv2.waitKey(0)

    # 使用open3d库绘制点云
    colorImage = o3d.geometry.Image(iml)
    depthImage = o3d.geometry.Image(depthMap)
    rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=1000.0,
                                                                     depth_trunc=np.inf)
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # fx = Q[2, 3]
    # fy = Q[2, 3]
    # cx = Q[0, 3]
    # cy = Q[1, 3]
    fx = config.cam_matrix_left[0, 0]
    fy = fx
    cx = config.cam_matrix_left[0, 2]
    cy = config.cam_matrix_left[1, 2]
    print(fx, fy, cx, cy)
    intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsics = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
    o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
    o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
    sys.exit(0)

    # 计算像素点的3D坐标（左相机坐标系下）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵

    # 构建点云--Point_XYZRGBA格式
    pointcloud = DepthColor2Cloud(points_3d, iml)

    # 显示点云
    # view_cloud(points_3d)
