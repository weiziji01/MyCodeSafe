#
# * 非极大值抑制操作
# coding=utf-8
# 导入python包
import numpy as np
import cv2

def non_max_suppression_slow(boxes, overlapThresh):
	# 如果输入为空，直接返回空列表
	if len(boxes) == 0:
		return []

	# 初始化列表索引
	pick = []

	# 获取边界框的坐标值
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# 计算边界框的区域大小并按照右下角的y坐标进行排序
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		# 获取索引列表中的最后一个索引，将索引值添加到所选索引的列表中，然后使用最后一个索引初始化禁止显示列表。
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]

		# 遍历索引列表中的所有索引
		for pos in range(0, last):
			# 获取当前的索引
			j = idxs[pos]

			# 查找边界框起点的最大（x，y）坐标和边界框终点的最小（x，y）坐标
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])

			# 计算边界框的宽和高
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)

			# 计算区域列表中计算的边界框和边界框之间的重叠率
			overlap = float(w * h) / area[j]

			# 如果它们具有较大的重叠率，则抑制掉它
			if overlap > overlapThresh:
				suppress.append(pos)

		# 从禁止显示列表中的索引列表中删除所有索引
		idxs = np.delete(idxs, suppress)

	# 返回选择的边界框
	return boxes[pick]

# 构建一个列表，其中包含将与其各自的边界框一起检查的图像
images = [
	("audrey.jpg", np.array([
	(12, 84, 140, 212),
	(24, 84, 152, 212),
	(36, 84, 164, 212),
	(12, 96, 140, 224),
	(24, 96, 152, 224),
	(24, 108, 152, 236)])),
	("bksomels.jpg", np.array([
	(114, 60, 178, 124),
	(120, 60, 184, 124),
	(114, 66, 178, 130)])),
	("gpripe.jpg", np.array([
	(12, 30, 76, 94),
	(12, 36, 76, 100),
	(72, 36, 200, 164),
	(84, 48, 212, 176)]))]

# 循环遍历所有的图像
for (imagePath, boundingBoxes) in images:
	# 读取图片并进行复制
	print ("[x] %d initial bounding boxes" % (len(boundingBoxes)))
	image = cv2.imread(imagePath)
	orig = image.copy()

	# 遍历每一个矩形框并绘制它们
	for (startX, startY, endX, endY) in boundingBoxes:
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

	# 应用非极大值抑制处理
	pick = non_max_suppression_slow(boundingBoxes, 0.3)
	print ("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))

	# 绘制处理之后的矩形框
	for (startX, startY, endX, endY) in pick:
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

	# 显示结果
	cv2.imshow("Original", orig)
	cv2.imshow("After NMS", image)
	cv2.waitKey(0)

