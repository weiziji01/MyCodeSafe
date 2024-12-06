import geopandas as gpd
from shapely.geometry import Polygon


def txt_to_shp(input_txt_path, output_shp_path):
    data = []
    with open(input_txt_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 10:
                continue  # 跳过格式不正确的行

            file_name = parts[0]
            confidence = float(parts[1])
            coords = [float(x) for x in parts[2:]]

            # 创建多边形
            polygon = Polygon(
                [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            )
            data.append(
                {"file_name": file_name, "confidence": confidence, "geometry": polygon}
            )

    # 使用GeoPandas创建GeoDataFrame
    gdf = gpd.GeoDataFrame(data, columns=["file_name", "confidence", "geometry"])

    # 保存为SHP文件
    gdf.to_file(output_shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"Shapefile saved to {output_shp_path}")


# 使用示例
input_txt = "/mnt/d/exp/hrsc_gf2/crop_test/Task1_ship.txt"  # 替换为你的输入TXT文件路径
output_shp = "/mnt/d/exp/hrsc_gf2/crop_test/ship.shp"  # 替换为你的输出SHP文件路径
txt_to_shp(input_txt, output_shp)
