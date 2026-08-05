---
name: yuque-access
description: "语雀知识库\"2026数模国赛备赛\"接入方式、API 关键参数与 skill 位置"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 9aec8e4d-ecef-4fc7-80c9-fb2aa93f8f57
  modified: 2026-08-05T16:13:53.710Z
---

队伍私有语雀知识库：https://www.yuque.com/amanannn/yyfbpg （48 篇文档，2026数模国赛备赛，repo id 72232834）

- token 由用户在 .env / 环境变量提供（不写入任何代码或仓库文件）。
- 接入 skill：github.com/flc1125/skills 的 skills/yuque（读文档/目录/创建更新文档）。Windows 设备需重新 clone 配置。
- API 要点（node fetch 直调）：header `X-Auth-Token`；
  - 创建文档：POST /api/v2/repos/72232834/docs，body {title, body, format:"markdown"}
  - 读 TOC：GET /api/v2/repos/72232834/toc
  - 挂目录（新版参数，重要）：PUT /api/v2/repos/72232834/toc，body {action:"appendNode", action_mode:"child", target_uuid:"", type:"DOC", title, doc_id} —— 旧参数 parent_uuid/sibling_uuid 已失效，prependNode 被禁用。
- 已上传语雀的文档：2023B题单刷计划（slug fsovfiqvpb3s8rm2）。
- 关联：[[project-conventions]]、[[b-preparation]]
