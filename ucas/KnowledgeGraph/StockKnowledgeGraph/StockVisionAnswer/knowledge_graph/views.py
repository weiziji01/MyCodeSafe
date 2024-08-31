import json
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# 读取知识图谱数据并缓存
file_path = os.path.join(settings.BASE_DIR, 'knowledge_graph/static/echarts_data.json')
with open(file_path, 'r', encoding='utf-8') as file:
    graph_data = json.load(file)
    nodes = graph_data['nodes']
    links = graph_data['links']

# 创建节点和关系的查询字典
node_dict = {node['name']: node for node in nodes}
link_dict = {(link['source'], link['target']): link['type'] for link in links}
link_dict.update({(link['target'], link['source']): link['type'] for link in links})

def get_graph_data(request):
    return JsonResponse(graph_data)

@csrf_exempt
def answer_question(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question')

        if '和' in question and '之间的关系' in question:
            entities = question.split('和')
            if len(entities) == 2:
                entity1 = entities[0].strip()
                entity2 = entities[1].split('之间的关系')[0].strip()
                entity1_id = node_dict[entity1]['id']
                # print(entity1_id)
                entity2_id = node_dict[entity2]['id']
                relationship = link_dict.get((entity1_id, entity2_id), "无关系")
                answer = f'{entity1}和{entity2}之间的关系是{relationship}。'
            else:
                answer = '请输入正确的实体名称。'
        elif any(node['name'] in question for node in nodes):
            entity = next((node['name'] for node in nodes if node['name'] in question), None)
            if entity:
                entity_info = node_dict.get(entity)
                answer = f'{entity}是{entity_info["category"]}。'  # 根据需求提供更多信息
            else:
                answer = '找不到相关实体。'
        else:
            answer = '对不起，我无法回答这个问题。'

        return JsonResponse({'answer': answer})
    return JsonResponse({'error': 'Invalid request'}, status=400)
