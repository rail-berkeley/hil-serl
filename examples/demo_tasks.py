
from absl import app, flags
import yaml

from rl_task import RLTask
from ros_robot_env.ros_robot_interface import RobotRosInterface


flags.DEFINE_string("tasks", "tasks.yaml", "all tasks config file path.")
flags.DEFINE_integer("seed", 40, "Random seed.")


class TaskScheduler:
    def __init__(self, tasks_file):
        self.tasks_file = tasks_file
        self.tasks = self.load_tasks()
        if len(self.tasks) == 0:
            raise ValueError("No tasks loaded")
        self.robot = RobotRosInterface(self.tasks[0].env.config)

    def load_tasks(self):
        tasks = []
        with open(self.tasks_file, 'r') as f:
            config_data = yaml.safe_load(f)
            if config_data and 'tasks' in config_data:
                for task_cfg in config_data['tasks']:
                    # 转换YAML数据为ExperimentConfig对象
                    task = RLTask(task_cfg)
                    tasks.append(task)
            print(f"成功加载 {len(tasks)} 个实验配置")
        return tasks

    def schedule_tasks(self):
        self.move_to_start()
        # self.robot.reset_robot_force_config()
        all_task_completed = True
        for task in self.tasks:
            result = task.run(self.robot)
            if result:
                task.env = None
                self.robot.set_force_config()
                task.log_current_pos()
                task.move_arm_to_safe_pos()
                print(f"task {task.name} completed successfully")
            else:
                print(f"task {task.name} failed")
                all_task_completed = False
                break
        self.move_to_start()

        if all_task_completed:
            for task in reversed(self.tasks):
                task.reset_object()

    def schedule_task(self, task):
        # Implement task scheduling logic here
        pass

    def move_to_start(self):
        pass

def main(_):
    scheduler = TaskScheduler(flags.FLAGS.tasks)
    scheduler.schedule_tasks()

if __name__ == "__main__":
    app.run(main)
