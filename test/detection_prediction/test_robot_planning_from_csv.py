#!/usr/bin/env python3
"""
测试脚本：从 CSV 文件读取前三列，生成机器人规划，并写回 CSV

输入：robot_planning_results.csv（前三列：任务、当前步骤、预测未来动作序列）
输出：更新 robot_planning_results.csv 的第四列（机器人下一步规划）
"""

import json
import csv
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# NOTE: translated from Chinese
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from train.task_planner import TaskGraphManager, EntropyPlanner


class ModifiedTaskGraphManager(TaskGraphManager):
    """
    修改版的 TaskGraphManager，支持从新的 taxonomy 文件路径读取
    """
    
    def __init__(self, taxonomy_path: str, task_name: str):
        """
        Args:
            taxonomy_path: 新的 taxonomy JSON 文件路径（直接包含任务字典）
            task_name: 当前视频对应的任务名称
        """
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # NOTE: translated from Chinese
            if isinstance(data, dict):
                self.taxonomy = data.get(task_name, {})
            else:
                self.taxonomy = {}
        
        if not self.taxonomy:
            raise ValueError(f"Task '{task_name}' not found in taxonomy")
        
        # NOTE: translated from Chinese
        self.id2node = {str(k): v for k, v in self.taxonomy.items()}
        self.name2node = {v.get('name', ''): v for k, v in self.taxonomy.items() if 'name' in v}
        self.name2id = {v.get('name', ''): str(k) for k, v in self.taxonomy.items() if 'name' in v}
        
        # NOTE: translated from Chinese (step)
        self.thread_map = self._build_thread_map()
        
        # NOTE: translated from Chinese
        self.and_gates = self._build_and_gates()


def parse_future_actions(future_str: str) -> List[str]:
    """
    解析预测未来动作序列字符串
    
    Args:
        future_str: CSV 中的未来动作序列字符串，可能是 "action1, action2, action3" 或 "none"
    
    Returns:
        动作列表
    """
    if not future_str or future_str.strip().lower() == 'none':
        return []
    
    # NOTE: translated from Chinese
    actions = [action.strip() for action in future_str.split(',') if action.strip()]
    return actions


def generate_robot_planning(
    csv_input_path: str,
    csv_output_path: str,
    taxonomy_path: str
):
    """
    从 CSV 读取数据，生成机器人规划，并写回 CSV
    
    Args:
        csv_input_path: 输入 CSV 文件路径
        csv_output_path: 输出 CSV 文件路径（可以与输入相同）
        taxonomy_path: taxonomy JSON 文件路径
    """
    # NOTE: translated from Chinese
    rows = []
    with open(csv_input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"读取了 {len(rows)} 行数据")
    
    # NOTE: translated from Chinese (cache)
    task_managers = {}
    planners = {}
    
    # NOTE: translated from Chinese
    results = []
    for i, row in enumerate(rows):
        task = row.get('任务', '').strip()
        current_step = row.get('当前步骤', '').strip()
        future_actions_str = row.get('预测未来动作序列', '').strip()
        
        if not task:
            print(f"警告：第 {i+1} 行任务名为空，跳过")
            results.append({
                'task': task,
                'current_step': current_step,
                'future_actions': future_actions_str,
                'robot_decision': 'ERROR: 任务名为空',
                'entropy': None
            })
            continue
        
        try:
            # NOTE: translated from Chinese
            if task not in task_managers:
                task_managers[task] = ModifiedTaskGraphManager(taxonomy_path, task)
                planners[task] = EntropyPlanner(task_managers[task])
            
            graph_mgr = task_managers[task]
            planner = planners[task]
            
            # NOTE: translated from Chinese (step)
            history_steps = [current_step] if current_step else []
            
            # NOTE: translated from Chinese (step)
            human_future = parse_future_actions(future_actions_str)
            
            # NOTE: translated from Chinese (predict)
            robot_history = []
            
            # NOTE: translated from Chinese
            best_robot_action, entropy_val = planner.decide(
                current_completed_steps=history_steps,
                human_future_steps=human_future,
                robot_history=robot_history
            )
            
            results.append({
                'task': task,
                'current_step': current_step,
                'future_actions': future_actions_str,
                'robot_decision': best_robot_action,
                'entropy': entropy_val
            })
            
            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(rows)} 行...")
                
        except Exception as e:
            error_msg = f'ERROR: {str(e)[:50]}'
            print(f"第 {i+1} 行处理失败: {error_msg}")
            results.append({
                'task': task,
                'current_step': current_step,
                'future_actions': future_actions_str,
                'robot_decision': error_msg,
                'entropy': None
            })
    
    # NOTE: translated from Chinese
    with open(csv_output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['任务', '当前步骤', '预测未来动作序列', '机器人下一步规划'])
        
        for r in results:
            writer.writerow([
                r['task'],
                r['current_step'],
                r['future_actions'],
                r['robot_decision']
            ])
    
    print(f"\n结果已保存到 {csv_output_path}")
    print(f"\n统计信息：")
    print(f"  总共处理了 {len(results)} 个样本")
    wait_count = sum(1 for r in results if r['robot_decision'] == 'Wait / None')
    error_count = sum(1 for r in results if r['robot_decision'].startswith('ERROR'))
    action_count = len(results) - wait_count - error_count
    print(f"  其中 'Wait / None': {wait_count} 个")
    print(f"  其中错误: {error_count} 个")
    print(f"  其中找到合法动作: {action_count} 个")
    
    # NOTE: translated from Chinese
    print(f"\n找到合法动作的样本示例（前10个）：")
    count = 0
    for r in results:
        if r['robot_decision'] != 'Wait / None' and not r['robot_decision'].startswith('ERROR'):
            print(f"  {r['task']} | {r['current_step']} | 机器人动作: {r['robot_decision']}")
            count += 1
            if count >= 10:
                break


def main():
    """主函数"""
    # NOTE: translated from Chinese (config)
    csv_file = project_root / 'robot_planning_results.csv'
    taxonomy_path = os.environ.get("TAXONOMY_JSON", str(project_root / "data" / "taxonomy" / "generated_taxonomy.json"))
    
    if not csv_file.exists():
        print(f"错误：找不到输入文件 {csv_file}")
        return
    
    if not os.path.exists(taxonomy_path):
        print(f"错误：找不到 taxonomy 文件 {taxonomy_path}")
        return
    
    print(f"输入文件: {csv_file}")
    print(f"Taxonomy 文件: {taxonomy_path}")
    print(f"输出文件: {csv_file} (覆盖输入文件)")
    print()
    
    # NOTE: translated from Chinese
    generate_robot_planning(
        csv_input_path=str(csv_file),
        csv_output_path=str(csv_file),
        taxonomy_path=taxonomy_path
    )


if __name__ == '__main__':
    main()

