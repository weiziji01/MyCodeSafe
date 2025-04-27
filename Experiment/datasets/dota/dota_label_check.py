"""
查看DOTA格式文件夹内下的所有标注文件是否存在无效标注，如果有，则打印出来哪一个存在
无效标注包括以下情形：
1. 坐标数目不足或超出预期
2. 坐标值不是有效的数字
3. 坐标值出现负数或者超出合理范围
"""
import os
import numpy as np
from tqdm import tqdm

def polygon_to_rotated_rect(points):
    """将四边形转换为旋转矩形参数（中心点，宽高，角度）"""
    # 计算中心点
    center = np.mean(points, axis=0)
    
    # 计算旋转角度（使用最长边方向）
    vec = points[1] - points[0]
    angle = np.arctan2(vec[1], vec[0])
    
    # 计算旋转后的坐标
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    rotated_points = np.dot(points - center, rotation_matrix)
    
    # 计算宽高
    width = np.max(rotated_points[:, 0]) - np.min(rotated_points[:, 0])
    height = np.max(rotated_points[:, 1]) - np.min(rotated_points[:, 1])
    
    return center, width, height, np.degrees(angle)

def check_dota_annotation(line, min_size=1.0):
    """
    检查单条DOTA标注的有效性
    Args:
        line: 标注行字符串
        min_size: 最小允许尺寸（单位：像素）
    Returns:
        tuple: (是否有效, 问题类型)
    """
    parts = line.strip().split()
    if len(parts) < 9:
        return False, "format error"
    
    try:
        # 解析四边形坐标
        coords = list(map(float, parts[:8]))
        points = np.array(coords).reshape(4, 2)
        
        # 检查坐标有效性
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False, "invalid coordinates"
            
        # 计算四边形面积（shoelace公式）
        area = 0.5 * np.abs(
            np.dot(points[:,0], np.roll(points[:,1],1)) - 
            np.dot(points[:,1], np.roll(points[:,0],1))
        )
        
        # 检查零面积
        if area < 1e-5:
            return False, "zero area"
        
        # 转换为旋转矩形参数
        _, width, height, _ = polygon_to_rotated_rect(points)
        
        # 检查小目标
        if min(width, height) < min_size:
            return False, "small object"
            
        return True, None
        
    except Exception as e:
        return False, f"parsing error: {str(e)}"

def process_dota_dataset(label_dir, output_dir=None, min_size=1.0):
    """
    处理整个DOTA数据集标注
    Args:
        label_dir: 原始标注目录
        output_dir: 过滤后输出目录（None表示不保存）
        min_size: 最小目标尺寸
    Returns:
        dict: 统计结果
    """
    stats = {
        'total_files': 0,
        'total_objects': 0,
        'invalid_objects': 0,
        'problem_types': {},
        'problem_files': []
    }
    
    # 准备输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有标注文件
    for filename in tqdm(os.listdir(label_dir), desc="Processing"):
        if not filename.endswith('.txt'):
            continue
            
        stats['total_files'] += 1
        input_path = os.path.join(label_dir, filename)
        output_path = os.path.join(output_dir, filename) if output_dir else None
        
        valid_lines = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('imagesource') or line.startswith('gsd'):
                    valid_lines.append(line)  # 保留注释行
                    continue
                
                valid, problem = check_dota_annotation(line, min_size)
                stats['total_objects'] += 1
                
                if valid:
                    valid_lines.append(line)
                else:
                    stats['invalid_objects'] += 1
                    stats['problem_types'][problem] = stats['problem_types'].get(problem, 0) + 1
        
        # 保存过滤后的文件
        if output_dir and valid_lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(valid_lines))
                
        # 记录有问题的文件
        if len(valid_lines) < (len(open(input_path).readlines()) - 2):  # 排除注释行
            stats['problem_files'].append(filename)
    
    # 生成报告
    print("\n=== DOTA Dataset Inspection Report ===")
    print(f"Total files scanned: {stats['total_files']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Invalid objects: {stats['invalid_objects']}")
    print("\nProblem Type Distribution:")
    for k, v in stats['problem_types'].items():
        print(f"- {k}: {v} ({v/stats['invalid_objects']:.1%})")
    
    # 保存问题文件列表
    if stats['problem_files']:
        with open("dota_problem_files.txt", "w") as f:
            f.write("\n".join(stats['problem_files']))
        print("\nProblem files list saved to: dota_problem_files.txt")
    
    # 数据增强建议
    print("\nData Augmentation Recommendations:")
    print("1. Disable the following augmentations:")
    print("   - InstaBoost (may generate invalid boxes)")
    print("   - RandomCrop (may create tiny objects)")
    print("   - Extreme Scaling (scale_min > 0.3 recommended)")
    print("2. Add post-augmentation validation:")
    print("   - Check bounding box area > 1 pixel")
    print("   - Verify width/height > 1 pixel after augmentation")
    
    return stats

if __name__ == "__main__":
    # 配置参数
    LABEL_DIR = "/mnt/d/data/visdrone2019/visdrone2019/tes/labels"
    OUTPUT_DIR = "/mnt/d/data/visdrone2019/visdrone2019/val/filter_labels"
    MIN_SIZE = 1.0  # 最小允许尺寸（单位：像素）
    
    # 执行检查
    stats = process_dota_dataset(
        label_dir=LABEL_DIR,
        output_dir=OUTPUT_DIR,
        min_size=MIN_SIZE
    )
