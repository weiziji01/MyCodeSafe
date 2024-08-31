"""
将json文件中存储的数据转换为echarts能够读取的格式
"""
import json

def parse_json_to_echarts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    nodes = []
    links = []
    
    for item in data:
        p = item['p']
        if p['start']['labels'][0] == '股票':
            start_node = {
                'id': str(p['start']['identity']),
                'name': str(p['start']['properties']['股票名称']),
                'value': str(p['start']['identity']),
                'category': str(p['start']['labels'][0]),
                # 'color' : '#87C2D2' # 蓝色
                'itemStyle':{'normal':{'color':'#87C2D2'}}
            }
        elif p['start']['labels'][0] == '概念':
            start_node = {
                'id': str(p['start']['identity']),
                'name': str(p['start']['properties']['概念名称']),
                'value': str(p['start']['identity']),
                'category': str(p['start']['labels'][0]),
                # 'color' : '#00FF00'  # 绿色
                'itemStyle':{'normal':{'color':'#00FF00'}}
            }
        elif p['start']['labels'][0] == '公告':
            start_node = {
                'id': str(p['start']['identity']),
                'name': str(p['start']['properties']['日期']),
                'value': str(p['start']['identity']),
                'category': str(p['start']['labels'][0]),
                # 'color' : '#0000FF'  # 蓝色
                'itemStyle':{'normal':{'color':'#0000FF'}}  
            }
        
        if p['end']['labels'][0] == '股票':
            end_node = {
                'id': str(p['end']['identity']),
                'name': str(p['end']['properties']['股票名称']),
                'value': str(p['end']['identity']),
                'category': str(p['end']['labels'][0]),
                # 'color' : '#F9E5E5'  # 粉色
                'itemStyle':{'normal':{'color':'#F9E5E5'}}
            }
        elif p['end']['labels'][0] == '概念':
            end_node = {
                'id': str(p['end']['identity']),
                'name': str(p['end']['properties']['概念名称']),
                'value': str(p['end']['identity']),
                'category': str(p['end']['labels'][0]),
                # 'color' : '#F08B9D'  # 粉色
                'itemStyle':{'normal':{'color':'#F08B9D'}}
            }
        elif p['end']['labels'][0] == '公告':
            end_node = {
                'id': str(p['end']['identity']),
                'name': str(p['end']['properties']['日期']),
                'value': str(p['end']['identity']),
                'category': str(p['end']['labels'][0]),
                # 'color' : '#C9B3D1'  # 紫色
                'itemStyle':{'normal':{'color':'#C9B3D1'}}
                
            }
        
        link = {
            'source': str(p['start']['identity']),
            'target': str(p['end']['identity']),
            'type': p['segments'][0]['relationship']['type']
        }
        
        nodes.append(start_node)
        nodes.append(end_node)
        links.append(link)
    
    nodes = {node['id']: node for node in nodes}.values()  # 去重
    return list(nodes), links

if __name__ == '__main__':
    nodes, links = parse_json_to_echarts('Pingan.json')

    with open('echarts_data.json', 'w', encoding='utf-8') as file:
        json.dump({'nodes': nodes, 'links': links}, file, ensure_ascii=False, indent=4)
