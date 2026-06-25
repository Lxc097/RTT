import random
import numpy as np
import matplotlib.pyplot as plt


class Env:
    """定义飞行空间、障碍物和无人机参数"""

    def __init__(self):
        # 飞行区域边界：[x_min, x_max, y_min, y_max]
        self.boundary = np.array([0.0, 100.0, 0.0, 100.0])

        # 圆形障碍物：[x, y, radius]
        self.obstacles = [
            [30, 30, 10],
            [60, 40, 15],
            [80, 70, 12],
            [20, 70, 8],
            [50, 80, 10]
        ]

        # 无人机参数
        self.max_step = 8.0          # 单次最大扩展步长
        self.robot_radius = 2.0      # 无人机机身半径
        self.goal_threshold = 5.0    # 接近终点阈值
        self.goal_sample_rate = 0.12 # 目标偏置概率


# -------------------- RRT 算法类 --------------------
class RRT:
    def __init__(self, env):
        self.env = env
        self.start = None
        self.goal = None
        self.tree = []

    def setup(self, start, goal):
        """设置起点、终点，并初始化随机树"""
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)

        # 检查起点和终点是否合法
        for name, point in [("起点", self.start), ("终点", self.goal)]:
            if not self.is_in_boundary(point):
                raise ValueError(f"{name}超出飞行区域边界！")

            if self.is_collision_with_obstacles(point):
                raise ValueError(f"{name}位于障碍物内或距离障碍物过近！")

        # 根节点为起点
        self.tree = [{"coord": self.start, "parent": -1}]

    def is_in_boundary(self, point):
        """检测点是否在无人机可安全飞行的边界内"""
        x_min, x_max, y_min, y_max = self.env.boundary
        r = self.env.robot_radius

        return (
            x_min + r <= point[0] <= x_max - r
            and y_min + r <= point[1] <= y_max - r
        )

    def random_sample(self):
        """随机采样；一定概率直接采样终点，提高搜索效率"""
        if random.random() < self.env.goal_sample_rate:
            return self.goal.copy()

        x_min, x_max, y_min, y_max = self.env.boundary
        r = self.env.robot_radius

        for _ in range(10000):
            x = random.uniform(x_min + r, x_max - r)
            y = random.uniform(y_min + r, y_max - r)
            point = np.array([x, y])

            if not self.is_collision_with_obstacles(point):
                return point

        raise RuntimeError("无法生成有效随机采样点，请检查障碍物设置。")

    def nearest_node(self, sample_point):
        """寻找树中距离采样点最近的节点"""
        distances = [
            np.linalg.norm(node["coord"] - sample_point)
            for node in self.tree
        ]

        nearest_idx = int(np.argmin(distances))
        nearest_coord = self.tree[nearest_idx]["coord"]

        return nearest_idx, nearest_coord

    def extend(self, nearest_coord, sample_point):
        """从最近节点朝采样点扩展一个最大步长"""
        direction = sample_point - nearest_coord
        distance = np.linalg.norm(direction)

        if distance == 0:
            return nearest_coord.copy()

        step = min(self.env.max_step, distance)

        return nearest_coord + direction / distance * step

    def is_collision_with_obstacles(self, point):
        """检测一个点是否与障碍物碰撞"""
        for obs_x, obs_y, obs_radius in self.env.obstacles:
            obs_center = np.array([obs_x, obs_y], dtype=float)

            # 障碍物半径 + 无人机半径 = 安全半径
            safe_radius = obs_radius + self.env.robot_radius

            distance = np.linalg.norm(point - obs_center)

            if distance <= safe_radius:
                return True

        return False

    def is_collision_with_path(self, start_point, end_point):
        """
        检测线段是否与圆形障碍物碰撞。
        使用“圆心到线段的最短距离”进行精确判断。
        """
        segment = end_point - start_point
        segment_length_sq = np.dot(segment, segment)

        for obs_x, obs_y, obs_radius in self.env.obstacles:
            obs_center = np.array([obs_x, obs_y], dtype=float)
            safe_radius = obs_radius + self.env.robot_radius

            # 计算障碍物圆心在线段上的投影位置
            if segment_length_sq == 0:
                closest_point = start_point
            else:
                t = np.dot(obs_center - start_point, segment) / segment_length_sq
                t = np.clip(t, 0.0, 1.0)
                closest_point = start_point + t * segment

            distance = np.linalg.norm(closest_point - obs_center)

            if distance <= safe_radius:
                return True

        return False

    def plan(self, max_iter=1500):
        """执行 RRT 路径规划"""

        # 先检查能否直接从起点飞往终点
        if not self.is_collision_with_path(self.start, self.goal):
            self.tree.append({
                "coord": self.goal.copy(),
                "parent": 0
            })
            print("起点和终点可直接连通，无需扩展随机树。")
            return self.extract_path()

        for iteration in range(max_iter):
            # 1. 随机采样
            sample = self.random_sample()

            # 2. 查找最近树节点
            nearest_idx, nearest_coord = self.nearest_node(sample)

            # 3. 向采样点扩展
            new_coord = self.extend(nearest_coord, sample)

            # 4. 检查新点边界和路径碰撞
            if not self.is_in_boundary(new_coord):
                continue

            if self.is_collision_with_path(nearest_coord, new_coord):
                continue

            # 5. 将新节点加入随机树
            self.tree.append({
                "coord": new_coord,
                "parent": nearest_idx
            })

            new_idx = len(self.tree) - 1
            distance_to_goal = np.linalg.norm(new_coord - self.goal)

            # 6. 检查是否可安全连接终点
            if distance_to_goal <= self.env.goal_threshold:
                # 已经精确到达终点时，不重复添加节点
                if distance_to_goal < 1e-8:
                    print(f"找到路径！迭代次数：{iteration + 1}")
                    return self.extract_path()

                # 接近终点时，仍要检测最后一段是否安全
                if not self.is_collision_with_path(new_coord, self.goal):
                    self.tree.append({
                        "coord": self.goal.copy(),
                        "parent": new_idx
                    })

                    print(f"找到路径！迭代次数：{iteration + 1}")
                    return self.extract_path()

        print("在最大迭代次数内未找到路径。")
        return None

    def extract_path(self):
        """从终点沿父节点回溯到起点，得到原始路径"""
        path = []
        current_idx = len(self.tree) - 1

        while current_idx != -1:
            node = self.tree[current_idx]
            path.append(node["coord"])
            current_idx = node["parent"]

        path.reverse()
        return np.array(path)

    def shortcut_path(self, raw_path):
        """
        路径剪枝优化：
        若两个非相邻节点之间能安全直连，则删除中间冗余节点。
        比三次样条插值更安全，不会使路径进入障碍物。
        """
        if raw_path is None or len(raw_path) < 3:
            return raw_path

        optimized_path = [raw_path[0]]
        i = 0

        while i < len(raw_path) - 1:
            j = len(raw_path) - 1

            # 从后向前寻找可直接连接的最远节点
            while j > i + 1:
                if not self.is_collision_with_path(raw_path[i], raw_path[j]):
                    break
                j -= 1

            optimized_path.append(raw_path[j])
            i = j

        return np.array(optimized_path)

    @staticmethod
    def path_length(path):
        """计算路径总长度"""
        if path is None or len(path) < 2:
            return 0.0

        return sum(
            np.linalg.norm(path[i] - path[i - 1])
            for i in range(1, len(path))
        )


# -------------------- 可视化函数 --------------------
def plot_results(env, rrt, raw_path, optimized_path):
    """绘制障碍物、随机树、原始路径和优化路径"""

    plt.figure(figsize=(10, 10))

    # 设置边界
    plt.xlim(env.boundary[0], env.boundary[1])
    plt.ylim(env.boundary[2], env.boundary[3])
    plt.grid(True, alpha=0.3)

    # 绘制障碍物与安全边界
    for i, obs in enumerate(env.obstacles):
        obs_x, obs_y, obs_radius = obs

        # 原始障碍物
        obstacle_circle = plt.Circle(
            (obs_x, obs_y),
            obs_radius,
            color="gray",
            alpha=0.7,
            label="Obstacle" if i == 0 else None
        )
        plt.gca().add_patch(obstacle_circle)

        # 加上无人机半径后的安全区域
        safe_circle = plt.Circle(
            (obs_x, obs_y),
            obs_radius + env.robot_radius,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Safety Boundary" if i == 0 else None
        )
        plt.gca().add_patch(safe_circle)

    # 绘制随机树
    for node in rrt.tree:
        if node["parent"] != -1:
            parent_coord = rrt.tree[node["parent"]]["coord"]

            plt.plot(
                [node["coord"][0], parent_coord[0]],
                [node["coord"][1], parent_coord[1]],
                "b-",
                linewidth=0.5,
                alpha=0.25
            )

    # 绘制原始路径
    if raw_path is not None:
        plt.plot(
            raw_path[:, 0],
            raw_path[:, 1],
            "r--",
            linewidth=2,
            label="Raw RRT Path"
        )

    # 绘制优化后的路径
    if optimized_path is not None:
        plt.plot(
            optimized_path[:, 0],
            optimized_path[:, 1],
            "g-",
            linewidth=3,
            label="Optimized Path"
        )

    # 绘制起点和终点
    plt.scatter(
        rrt.start[0],
        rrt.start[1],
        color="green",
        s=120,
        marker="o",
        label="Start",
        zorder=5
    )

    plt.scatter(
        rrt.goal[0],
        rrt.goal[1],
        color="red",
        s=120,
        marker="x",
        label="Goal",
        zorder=5
    )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("UAV RRT Path Planning (2D)")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 固定随机种子，便于复现实验结果
    random.seed(42)
    np.random.seed(42)

    # 1. 创建环境
    env = Env()

    # 2. 创建 RRT 规划器
    rrt = RRT(env)

    # 3. 设置起点和终点
    start = [10, 10]
    goal = [90, 90]
    rrt.setup(start, goal)

    # 4. 执行路径规划
    raw_path = rrt.plan(max_iter=1500)

    # 5. 路径优化
    if raw_path is not None:
        optimized_path = rrt.shortcut_path(raw_path)

        print(f"原始路径节点数：{len(raw_path)}")
        print(f"优化路径节点数：{len(optimized_path)}")
        print(f"原始路径长度：{rrt.path_length(raw_path):.2f} m")
        print(f"优化路径长度：{rrt.path_length(optimized_path):.2f} m")
    else:
        optimized_path = None
        print("未找到可行路径！")

    # 6. 显示结果
    plot_results(env, rrt, raw_path, optimized_path)