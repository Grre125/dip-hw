import cv2
import numpy as np
import gradio as gr
from scipy.interpolate import Rbf

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None


# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
'''
evt.index 提供了用户点击的坐标 (x, y)。
偶数次点击作为源点（蓝色），奇数次点击作为目标点（红色）。
在图像上用圆圈标记点击的点，并在成对的点之间画出绿色箭头，表示映射关系。
最终返回标记了点的图像供用户查看。
'''
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image






# 执行仿射变换

# def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, epsilon=1e-8,func='multiquadric'):
def point_guided_deformation(im, psrc, pdst):
    # todo: FILL: 基于MLS or RBF 实现 image warping  --- 这个还不行
    # h, w, c = image.shape
    # warped_image = np.zeros_like(image)
    #
    #
    # # 如果没有足够的控制点，直接返回原图
    # if len(source_pts) < 3:
    #     return image
    #
    # # 提取源点的 x, y 坐标
    # src_x, src_y = source_pts[:, 0], source_pts[:, 1]
    # dst_x, dst_y = target_pts[:, 0], target_pts[:, 1]
    #
    # # 基于 RBF 插值构建映射关系
    # rbf_x = Rbf(src_x, src_y, dst_x, function=func, epsilon=epsilon)
    # rbf_y = Rbf(src_x, src_y, dst_y, function=func, epsilon=epsilon)
    #
    # # 创建网格，包含图像中的所有像素点
    # grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # grid_x_flat = grid_x.flatten()
    # grid_y_flat = grid_y.flatten()
    #
    # # 通过 RBF 计算每个像素的新位置
    # new_x = rbf_x(grid_x_flat, grid_y_flat).reshape(h, w)
    # new_y = rbf_y(grid_x_flat, grid_y_flat).reshape(h, w)
    #
    # # 将新的坐标点限制在图像的边界内
    # new_x = np.clip(new_x, 0, w - 1).astype(np.float32)
    # new_y = np.clip(new_y, 0, h - 1).astype(np.float32)
    #
    # # 使用 OpenCV 的 remap 函数进行插值，将图像映射到新的坐标
    # warped_image = cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)
    #
    # return warped_image

    import numpy as np

    def point_guided_deformation(image, source_points, target_points):
        """
        使用控制点和目标点进行图像变形（基于RBF或MLS思想）。

        参数:
        - image: 原始图像，形状为 (height, width, channels)。
        - source_points: 控制点的源位置 (psrc)，形状为 (n, 2)。
        - target_points: 控制点的目标位置 (pdst)，形状为 (n, 2)。

        返回:
        - 变形后的图像，形状与原始图像相同。
        """
        # 获取图像的高度、宽度和通道数
        height, width, channels = image.shape
        num_points = len(source_points)

        # 初始化变形后的图像为白色
        warped_image = np.ones((height, width, channels), dtype=np.uint8) * 255

        # 如果没有控制点，则返回空白图像
        if num_points == 0:
            return warped_image

        # 交换 source_points 和 target_points 的列（因为图像坐标是 (y, x) 而非 (x, y)）
        source_points = source_points[:, [1, 0]]
        target_points = target_points[:, [1, 0]]

        # 计算源点和目标点之间的距离平方和
        total_distance = np.sum(np.linalg.norm(target_points - source_points, axis=1) ** 2)

        # 计算距离矩阵 D
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                distance_matrix[i, j] = 1 / (np.sum((target_points[i] - target_points[j]) ** 2) + total_distance)

        # 计算变换矩阵 B
        transform_matrix = np.linalg.solve(distance_matrix, source_points - target_points)

        # 创建网格，包含图像的每个像素点
        x_grid, y_grid = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
        delta_x = np.zeros((height, width))
        delta_y = np.zeros((height, width))

        # 通过 RBF 或 MLS 算法计算每个像素的新位置
        for k in range(num_points):
            influence = 1 / ((x_grid - target_points[k, 0]) ** 2 + (y_grid - target_points[k, 1]) ** 2 + total_distance)
            delta_x += transform_matrix[k, 0] * influence
            delta_y += transform_matrix[k, 1] * influence

        # 将计算出的偏移量加回到原始坐标中，并将坐标转换为整数
        new_x = np.round(delta_x + x_grid).astype(int)
        new_y = np.round(delta_y + y_grid).astype(int)

        # 确保坐标在有效范围内
        new_x = np.clip(new_x, 1, height)
        new_y = np.clip(new_y, 1, width)

        # 逐像素进行映射，将原图像的像素映射到新图像
        for i in range(height):
            for j in range(width):
                if 1 <= new_x[i, j] <= height and 1 <= new_y[i, j] <= width:
                    warped_image[i, j, :] = image[new_x[i, j] - 1, new_y[i, j] - 1, :]
                else:
                    warped_image[i, j, :] = [0, 0, 0]  # 超出范围的像素设置为黑色

        # 转置图像以恢复原始顺序 (width, height, channels)
        warped_image = np.transpose(warped_image, (1, 0, 2))

        return warped_image


def run_warping():
    global points_src, points_dst, image  ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image






# def run_warping():
#     global points_src, points_dst, image # fetch global variables
#
#     warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))
#
#     return warped_image

def run_warping():
    global points_src, points_dst, image

    # 确保点集的数量相等且不为空
    if len(points_src) == len(points_dst) and len(points_src) > 0:
        warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))
        return warped_image
    else:
        return image  # 如果点不匹配，则返回原图

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
