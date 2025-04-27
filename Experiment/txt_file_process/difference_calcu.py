"""
计算两个txt文件中共同项数据的差值，存储在一个新的txt文件中
原有txt文件内的格式：
----            |---
apple,100       |apple,90
banana,200      |banana,180
orange,50       |grape,300
----            |----
生成的txt文件内的格式：
共同项,文件1值,文件2值,差值
apple,100.0,90.0,10.0
banana,200.0,180.0,20.0
"""
def process_files(file1_path, file2_path, output_path):
    # 读取第一个txt文件 {关键字: 值}
    dict1 = {}
    with open(file1_path, 'r') as f1:
        for line in f1:
            line = line.strip()
            if line:
                try:
                    key, value = line.split(',')
                    dict1[key] = float(value)  # 假设第二列是数值类型
                except (ValueError, TypeError):
                    continue  # 跳过格式错误行

    # 读取第二个txt文件 {关键字: 值}
    dict2 = {}
    with open(file2_path, 'r') as f2:
        for line in f2:
            line = line.strip()
            if line:
                try:
                    key, value = line.split(',')
                    dict2[key] = float(value)
                except (ValueError, TypeError):
                    continue

    # 计算共同项差值
    common_keys = set(dict1.keys()) & set(dict2.keys())
    results = []
    for key in common_keys:
        value1 = dict1[key]
        value2 = dict2[key]
        diff = value2 - value1
        results.append(f"{key},{value1},{value2},{diff}")

    # 写入结果到新文件
    with open(output_path, 'w') as f_out:
        f_out.write("图片名,s2a_sca,sae_sca,差值\n")  # 写入表头
        f_out.write("\n".join(results))

# 使用示例
process_files(
    file1_path="/mnt/d/learning/空天院/论文/01-paper1/SCA计算/visdrone/sca_record/s2anet_sca_visdrone.txt",
    file2_path="/mnt/d/learning/空天院/论文/01-paper1/SCA计算/visdrone/sca_record/saedet_sca_visdrone.txt",
    output_path="/mnt/d/learning/空天院/论文/01-paper1/SCA计算/visdrone/sca_record/diff_result.txt"
)