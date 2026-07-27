# 编程手（队长）Claude Code 配置指南

> 适配角色：数学建模 **编程手 & 队长** — 负责代码实现、数据处理、算法调试、可视化、团队管理
> 你的工具链：Python (NumPy/SciPy/Pandas/Matplotlib) / MATLAB + LaTeX + Git

---

## 一、Claude Code 能帮你做什么

| 场景 | Claude Code 能力 |
|------|------------------|
| 实现预测/评价/优化模型 | 写代码、调参、向量化优化 |
| 数据预处理与可视化 | 生成数据处理管线、高质量图表 |
| 调试模型结果异常 | 系统性定位 Bug，对比理论值与计算值 |
| 批量生成代码模板 | 一次性产出多套模板代码 |
| 管理团队代码仓库 | Git 操作、代码审查、文档生成 |
| 4天实战中快速响应 | Day1-3 高强度的编码、调优、改图 |

---

## 二、推荐安装的插件/Skills

### 必装（核心生产力）

**1. Superpowers 插件包**

包含多个子技能，一键安装：
```bash
claude plugins install superpowers
```

你高频使用的子技能：
- `brainstorming` — 拿到题目后，先和 Claude 头脑风暴算法方案，再动手写
- `test-driven-development` — 先写测试用例验证模型逻辑，再实现（避免模型写错返工）
- `systematic-debugging` — 结果不对时，按流程排查（不瞎改）
- `verification-before-completion` — 提交代码前强制验证输出是否合理
- `writing-plans` + `executing-plans` — 复杂实现先出计划，再分步执行
- `subagent-driven-development` — 多个独立模块并行开发（如同时写预测模型和评价模型）
- `using-git-worktrees` — 实验性算法在隔离环境中尝试，不影响主代码

**2. Claude HUD 插件**

```bash
claude plugins install claude-hud
```

在终端底部常驻状态栏，实时显示当前任务、模型、token 用量。你的 4 天实战节奏很紧，HUD 能帮你快速感知工作状态。

**3. Planning with Files 插件**

```bash
claude plugins install planning-with-files
```

每个专题训练开始前，用 `plan` 命令生成 task_plan.md / findings.md / progress.md，把 4 天的编码任务结构化跟踪。

### 可选（按需）

**4. Ralph Loop 插件**

```bash
claude plugins install ralph-loop
```

当你需要持续监控长时间运行的任务（如蒙特卡洛仿真跑 10 万次），让 Claude 定期检查进度。

---

## 三、settings.json 配置

在你的项目目录 `D:\虚拟C盘\数学建模培训\.claude\settings.local.json` 中配置：

```json
{
  "permissions": {
    "allow": [
      "Bash(python *)",
      "Bash(python3 *)",
      "Bash(pip *)",
      "Bash(pip3 *)",
      "Bash(conda *)",
      "Bash(matlab *)",
      "Bash(git *)",
      "Bash(mkdir *)",
      "Bash(code *)",
      "Bash(pdftotext *)",
      "Bash(ls *)",
      "Bash(cd *)",
      "Bash(cp *)",
      "Bash(mv *)",
      "Bash(rm *)",
      "Bash(cat *)",
      "Bash(wc *)"
    ],
    "deny": []
  },
  "enableAllProjectMcpServers": true
}
```

> **说明**：授权 Claude 直接执行 Python、Git、文件操作等命令，避免每次都要手动确认，节省 4 天实战中的宝贵时间。

### 如果使用 MATLAB

```json
{
  "permissions": {
    "allow": [
      "Bash(matlab -batch *)",
      "Bash(matlab -nodisplay *)"
    ]
  }
}
```

---

## 四、推荐 Hook 配置

在 `.claude/settings.local.json` 中添加 hooks，让 Claude 在关键节点自动执行检查：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "python -m py_compile \"$CLAUDE_TOOL_OUTPUT\" 2>/dev/null && echo '✅ Python syntax OK' || echo '❌ Python syntax error'"
          }
        ]
      }
    ]
  }
}
```

> 每次 Write Python 文件后自动检查语法，防止低级错误带到 Day4 论文里。

---

## 五、项目 CLAUDE.md 配置

在项目根目录 `D:\虚拟C盘\数学建模培训\CLAUDE.md` 创建：

```markdown
# 数学建模国赛 — 编程手工作区

## 项目背景
2026 年高教社杯全国大学生数学建模竞赛备赛。72 小时竞赛（9.10 18:00 - 9.13 20:00）。

## 编程环境
- Python 3.x (NumPy, SciPy, Pandas, Matplotlib, Scikit-learn)
- MATLAB (可选)
- LaTeX (论文编译)

## 项目结构
- /models/     — 各专题模型代码
- /data/       — 数据集
- /utils/      — 通用工具函数
- /templates/  — 代码模板
- /output/     — 图表输出
- /notes/      — 学习笔记、操作截图

## 代码规范
- 函数添加中文注释说明用途和参数
- 每个模型提供独立的 demo 脚本
- 图表导出为高清 PNG/EPS（方便论文手直接使用）
- 使用 Git 管理所有代码，每次训练后打 tag

## 常用指令
- 写代码前先 brainstorm 算法方案
- 实现后立即写测试验证
- 出图时确认坐标轴标签、单位、图例完整
```

---

## 六、快速上手流程

### 第一步：安装插件（一次性）

```bash
claude plugins install superpowers
claude plugins install claude-hud
claude plugins install planning-with-files
claude plugins install ralph-loop
```

### 第二步：配置 HUD

```bash
/claude-hud:setup
```

按提示选择布局和显示项。

### 第三步：配置项目

把上面的 settings.local.json 和 CLAUDE.md 放到对应位置，重启 Claude Code。

### 第四步：日常使用模式

```
# 开始一个专题训练
/plan     → 生成训练任务计划

# 实现新模型
"帮我用 Python 实现灰色预测模型 GM(1,1)，要有完整注释和 demo"

# 调试
"这段代码的预测结果和理论值偏差很大，帮我排查" → Claude 自动触发 systematic-debugging

# 出图
"把这个预测结果画成对比图，要求：中文字体、图例、坐标轴标签、保存为 EPS"

# 提交前
"验证所有模型代码的输出是否合理" → Claude 自动触发 verification-before-completion

# 代码审查
/code-review → 检查代码质量和潜在 bug
```

---

## 七、4 天实战节奏 × Claude Code 使用策略

| 天数 | 你的任务 | Claude Code 用法 |
|------|----------|------------------|
| **Day1** | 审题 + 评估计算可行性 | `brainstorming` 讨论算法方案，列出备选和风险评估 |
| **Day2** | 主力编码实现 | `test-driven-development` 先写测试再实现；`subagent-driven-development` 并行开发独立模块 |
| **Day3** | 优化调参 + 出图 | 批量调参循环；要求 Claude 生成多组对比图 |
| **Day4** | 配合论文手整理代码附录 | 自动提取代码关键段、生成伪代码、整理注释 |

---

## 八、国赛 72 小时的 Claude Code 策略

> ⚠️ 正式比赛时网络可能受限，提前确认 Claude Code 的可用性。以下为理想情况：

| 时间段 | 重点 | 用法 |
|--------|------|------|
| 前 6 小时 | 选题 + 审题 | brainstorm 所有题目，快速评估团队可行性 |
| 6-24 小时 | 建模 + 初版代码 | 并行实现多个候选模型，快速淘汰不合适的 |
| 24-48 小时 | 优化 + 批量出图 | 调参自动化、生成全量图表 |
| 48-66 小时 | 论文配合 | 提取代码说明、生成流程图伪代码 |
| 66-72 小时 | 查漏补缺 | 最终验证所有输出、检查数据一致性 |

---

> 📌 **一句话**：Claude Code 是你的 AI 结对编程搭档 — 它帮你写得更快、调得更准，但模型选择、算法判断、创新点的决策权在你手里。
