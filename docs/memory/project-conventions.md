---
name: project-conventions
description: 项目代码与文档约定、Git 规范、已知缺陷清单
metadata: 
  node_type: memory
  type: project
  originSessionId: 9aec8e4d-ecef-4fc7-80c9-fb2aa93f8f57
  modified: 2026-08-05T16:13:55.598Z
---

仓库：github.com/amanannn/math-modeling-2026（公开，分支 main，已删 master）

- 代码注释：简短中文，重要部分一句；模型 demo 在 models/，学习笔记在 notes_plain/（文件名带负责人）。
- 提交规范：feat:/docs: 前缀 + 中文描述。
- 隐私：README 与公开门面不出现队员姓名；语雀 token 绝不入仓。
- 已知缺陷（答辩风险）：
  - 2023C 仓库 `TOPSIS+熵权法` 脚本实为等权 [0.5,0.5]，无熵权法实现
  - 2024A Question4.py：rotate_point 参数顺序错误（调头段板长约束破坏 3.6 倍）、396 行变量 mid 未定义、corner() 垂直板除零
- 关联：[[b-preparation]]、[[team-info]]
