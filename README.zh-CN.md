# Humanoid Agent World Simulation

**一个让“微观人类行为”涌现出“宏观世界结果”的人类响应模拟引擎。**

[English README](README.md)

如果油价暴涨 100%、疫情爆发、龙头企业裁员、或者一家巧克力公司被曝光使用童工，接下来会发生什么？

这个项目要做的，不是普通的多智能体聊天 demo，而是一个真正的 **人类行为 + 经济涟漪** 引擎：

- 每个 agent 都有持续存在的情绪、人格偏置、记忆、关系、债务压力、阵营归属和私人叙事
- 这些 agent 会像真实人类一样，根据自己的处境做出不同反应
- 这些微观决策会沿着企业、机构、社区、关系网络和信息传播链不断扩散
- 最终形成可衡量的宏观结果，比如消费者信心变化、市场压力、行业分化、社会情绪和冲突升级

这个系统的核心不是“让所有 agent 每一步都调用一次 LLM”，而是构建一个 **可扩展、可解释、可验证** 的社会世界模拟底座。

## 核心架构

### Heart Engine

负责 agent 的“内心世界”：

- 情绪状态
- 人格与应对风格
- 创伤与脆弱性
- 事件主观解释
- 动机冲突
- 掩饰机制
- 解释型记忆
- 持续中的个人故事线

这决定了一个 agent 不是“同一个模型换个名字”，而是真正拥有不同的心理结构和行为偏差。

### Ripple Engine

负责把微观反应变成宏观传播：

- 关系变化
- 信息扩散
- 谣言传播
- 阵营和联盟形成
- 债务压力
- 群体冲突
- boycotts、hearing、leak、mutual aid 等动态事件
- 跨行业、跨群体的宏观指标聚合

这部分让世界不再是静态背景，而是一个会被 agent 行为持续重写的系统。

### LLM Agency

LLM 在这里不是整个世界的底层，而是“关键角色的主动意识层”。

项目里已经有：

- 高显著性 agent 筛选
- LLM 决策 packet 预构建
- 自定义 world snapshot
- what-if shock 注入
- 3D 世界前端和单个 agent 检视面板

目标是：

- 大部分 agent 走低成本确定性模拟
- 关键 CEO、官员、管理者、组织者、金融参与者在关键时刻获得 LLM 决策能力

这样既保留规模，也保留关键场景的人类复杂度。

## 当前最强卖点

这不是一个玩具模拟器。

这是一个 **从微观人类行为中涌现宏观社会与经济结果** 的系统。

它和普通 LLM-only agent 世界最大的区别在于：

- 普通系统强在“会说话”
- 这个系统强在“有持续状态、有因果链、有可追踪的后果”

你不只是能看到 agent 说了什么，还能追问：

- 他为什么这样做？
- 他在害怕什么？
- 他在隐藏什么？
- 哪段关系因此改变了？
- 这件小事最后为什么会影响整个社区、行业甚至市场？

## 当前验证结果

### 历史事件宏观验证

仓库里已经有历史事件验证框架，会把真实世界冲击注入模拟世界，再把宏观结果与历史规律对照。

当前结果：

- 历史宏观验证总体准确率：**83.6%**
- 方向判断准确率：**86.7%**
- 量级检验通过率：**100%**

分事件结果：

- 2008 油价冲击：**100.0%**
- 2008 银行恐慌 / 雷曼危机：**84.6%**
- COVID-19 初期冲击：**83.3%**
- 品牌丑闻 / boycott 模式：**75.0%**
- 军事危机 / 核危机模式：**75.0%**

对应 artifact：

- [`examples/artifacts/historical_validation.json`](examples/artifacts/historical_validation.json)

### 个体差异验证

在 5 选 1 的角色识别盲测中，系统达到：

- **90% top-1**
- 随机水平只有 **20%**

对应 artifact：

- [`artifacts/character_identity_blind_test_pass2_fixed_20260322.json`](artifacts/character_identity_blind_test_pass2_fixed_20260322.json)

这说明 agent 已经开始具备可区分的个体特征，而不再只是“同一个人换个名字”。

### 大型世界运行

在当前 50 天港口危机世界中，系统跑出了：

- **300 agents**
- **1,569 个已触发事件**
- **1,542 个 ripple events**
- **5,952 次互动**
- **7,567 对最终关系**

对应 artifact：

- [`artifacts/heatwave_harbor_50d_pass2_fixed_20260322.json`](artifacts/heatwave_harbor_50d_pass2_fixed_20260322.json)

## 这个系统能做什么

### 1. 金融与经济推演

- 油价暴涨会怎样传导到运输、消费、就业和信心？
- 银行恐慌会怎样改变储户、企业主、白领和政府机构的行为？
- 刺激政策、裁员、加息、供应链冲击会怎样在社会层面扩散？

### 2. 企业与品牌危机模拟

- 如果一家消费品牌被曝光使用童工，会怎样影响消费者、员工、媒体和监管？
- boycott 会不会扩大？
- 哪些群体会首先转移购买？
- 哪些叙事会主导舆论？

### 3. 政策与治理测试

- 疫情、宵禁、公共卫生政策、能源政策发布后，社会会怎样变化？
- 哪些群体会服从，哪些群体会反弹？
- 信息传播会如何改变制度信任？

### 4. 游戏与数字世界

- 构建真正“会记仇、会结盟、会演化”的 NPC 社会
- 玩家不是触发脚本，而是在一个会真实反应的社会中行动

## 快速开始

### 1. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn
cd frontend && npm install && cd ..
```

### 2. 启动前后端

```bash
./start.sh
```

默认地址：

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`

### 3. 打开 3D 世界

访问：

- `http://localhost:3000/#/world`

### 4. 注入现实冲击

例如：

```bash
curl -X POST http://127.0.0.1:8000/api/world/what_if \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "heatwave_harbor",
    "days": 14,
    "information": "oil prices surge 100%",
    "llm_samples": 0
  }'
```

可用接口：

- `GET /api/world/snapshot`
- `POST /api/world/snapshot/custom`
- `POST /api/world/what_if`

## 关键目录

| 路径 | 作用 |
| --- | --- |
| `examples/learned_brain/world_sim/` | 世界模拟核心 |
| `examples/learned_brain/world_sim/world.py` | 世界 tick 循环与事件执行 |
| `examples/learned_brain/world_sim/world_agent.py` | agent 内部状态 |
| `examples/learned_brain/world_sim/world_information.py` | 现实冲击注入 |
| `examples/learned_brain/world_sim/info_propagation.py` | 信息传播 |
| `examples/learned_brain/world_sim/macro_aggregator.py` | 宏观指标聚合 |
| `examples/learned_brain/world_sim/eval/historical_validation.py` | 历史事件验证 |
| `api_server.py` | FastAPI 后端 |
| `frontend/src/world-viewer/` | 3D 世界前端 |

## 项目方向

这个项目最终想做的，不只是“更真实的 NPC”。

而是一个可扩展的人类响应模拟底座：

- 用真实的人类行为逻辑驱动世界
- 支持从微观行为推演宏观社会和经济结果
- 能注入真实世界冲击并追踪传播链
- 可服务于金融、政策、品牌、游戏和更广泛的行业场景

一句话概括：

> **Humanoid Agent World Simulation 不是让 agent 看起来像人，而是让世界像真的会被人改变一样运转。**

## 当前仍在补强的部分

- 更强的现实数据校准
- 更细分行业的预测模型
- 更完善的高显著性 LLM 决策执行
- 更大规模世界上的稳定性与性能

## License

MIT
