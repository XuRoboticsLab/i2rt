# 奇异鲁棒 IK：数学原理

## 1. 问题背景

对于六自由度机械臂，微分 IK 的核心是在每一步求解：

$$
\dot{q} = J^+ \, \dot{x}
$$

其中 $J \in \mathbb{R}^{6 \times 6}$ 是末端雅可比矩阵，$\dot{x}$ 是末端速度误差，$J^+$ 是伪逆。

当机械臂接近**奇异构型**时，$J$ 的最小奇异值 $\sigma_{\min} \to 0$，导致：

$$
J^+ = (J^T J)^{-1} J^T \implies \|J^+\| \to \infty
$$

即使 $\dot{x}$ 很小，$\dot{q}$ 也会爆炸——关节速度无界，机械臂产生剧烈跳变。

---

## 2. Levenberg-Marquardt 阻尼（第一层）

### 原理

将伪逆替换为**阻尼最小二乘**：

$$
\dot{q} = J^T \left( J J^T + \lambda I \right)^{-1} \dot{x}
$$

其中 $\lambda > 0$ 是阻尼系数。这等价于求解带正则化的优化问题：

$$
\min_{\dot{q}} \; \|\dot{x} - J\dot{q}\|^2 + \lambda \|\dot{q}\|^2
$$

### 效果

对 $J$ 做 SVD：$J = U \Sigma V^T$，则：

$$
\dot{q} = V \, \text{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) U^T \dot{x}
$$

- 当 $\sigma_i \gg \sqrt{\lambda}$：$\dfrac{\sigma_i}{\sigma_i^2 + \lambda} \approx \dfrac{1}{\sigma_i}$，接近真实伪逆，精度无损
- 当 $\sigma_i \to 0$（奇异方向）：$\dfrac{\sigma_i}{\sigma_i^2 + \lambda} \approx \dfrac{\sigma_i}{\lambda} \to 0$，关节速度被压制

$\lambda$ 越大，奇异鲁棒性越强，但末端跟踪精度略有下降。本实现默认 $\lambda = 10^{-3}$（原始值 $10^{-4}$）。

---

## 3. 关节中心吸引力（第二层）

### 动机

LM 阻尼只限制关节速度的幅值，不限制方向——IK 仍可能漂向关节限位，而关节极限附近恰好是奇异高发区。

### 方法

设关节限位为 $[q_{\min,i},\, q_{\max,i}]$，定义关节中心：

$$
q_c = \frac{q_{\min} + q_{\max}}{2}
$$

在每次微分 IK 迭代后，对关节角施加一个小幅拉力：

$$
q \leftarrow q + \alpha \left( q_c - q \right)
$$

其中 $\alpha = 10^{-3}$ 是增益。

### 与 PostureTask 的等价性

这等价于在 QP 中添加一个二次正则项（PostureTask）：

$$
\min_{\dot{q}} \; \|e_{ee}\|^2 + w \|q - q_c\|^2
$$

主任务（末端误差 $e_{ee}$）权重为 1，姿态正则权重 $w \ll 1$。

- 末端可达时：两项同时趋零，末端精度不受影响
- 多解区域（奇异邻域）：QP 自然选择靠近 $q_c$ 的解，远离关节极限

---

## 4. 关节空间速度限幅（第三层）

### 动机

前两层在 IK 内部工作，但如果目标位姿突变（例如 Twist 消息堆积），IK 可能在一个控制周期内求出合法但跳变很大的解。

### 方法

设上一周期末端关节角为 $q_{\text{prev}}$，IK 新解为 $q_{\text{sol}}$，定义：

$$
\Delta q = q_{\text{sol}} - q_{\text{prev}}
$$

若 $\|\Delta q\|_\infty > \Delta q_{\max}$，则按比例缩放：

$$
q_{\text{cmd}} = q_{\text{prev}} + \Delta q \cdot \frac{\Delta q_{\max}}{\|\Delta q\|_\infty}
$$

这保证每个关节每周期的位移不超过 $\Delta q_{\max}$（默认 $0.15\,\text{rad}$，50 Hz 下对应约 $7.5\,\text{rad/s}$）。

---

## 5. 三层协同

$$
\underbrace{(JJ^T + \lambda I)^{-1}}_{\text{层1：LM阻尼}}
\xrightarrow{\text{迭代}} \;
\underbrace{q \leftarrow q + \alpha(q_c - q)}_{\text{层2：关节中心吸引}}
\xrightarrow{\text{解后裁剪}} \;
\underbrace{\|\Delta q\|_\infty \leq \Delta q_{\max}}_{\text{层3：速度限幅}}
$$

| 层 | 防护的失效模式 | 代价 |
|---|---|---|
| LM 阻尼 | Jacobian 降秩时关节速度爆炸 | 末端轨迹略有偏差 |
| 关节中心吸引 | IK 漂向关节极限（奇异高发区） | 极小的关节偏置 |
| 速度限幅 | 目标突变导致单步大跳变 | 末端跟踪有短暂滞后 |

三层叠加后，机械臂在正常工作空间内的任意运动均保持平滑可控，且不依赖额外的运动学库。

---

## 参考

- Nakamura & Hanafusa (1986). *Inverse kinematic solutions with singularity robustness*
- Wampler (1986). *Manipulator inverse kinematic solutions based on vector formulations*
- Buss (2004). *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least Squares methods*
