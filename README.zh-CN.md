# Humanoid Agent World Simulation

[English README](README.md)

> 一个面向投资、政策、企业战略和虚拟世界构建的 AI 世界模型。

Humanoid Agent World Simulation 不是普通的多智能体 demo。它试图回答一个更大的问题：

如果我们把真实世界里的情绪、组织、政策、供需关系、恐慌、信任和权力结构都放进同一个模拟系统里，能不能从微观行为中自然涌现出可信的宏观结果？

这个仓库就是朝这个方向做出来的原型系统。

你给它一个现实冲击，比如：

- 油价暴涨
- 银行体系恐慌
- 疫情暴发
- 品牌丑闻
- 激进加息或刺激政策

系统中的 agent 不会像模板机器人一样同步反应，而是会基于各自的性格、情绪、创伤、债务压力、社会关系和制度位置做出不同选择。然后这些选择会沿着公司、机构、消费者和政策网络一路传导，最终形成可观察的宏观变化。

## 为什么这件事重要

大多数模拟系统的问题在于两点：

- 要么只会给你宏观数字，但解释不了这些结果是怎么产生的
- 要么完全依赖 LLM，成本太高，规模上不去

这个项目采取的是混合架构：

- 确定性引擎负责每个 tick 的世界推进
- LLM 只在关键决策者和高价值时刻介入

结果是，它既有解释性，也有扩展性。

这让它非常适合未来的高价值场景：

- 投资机构做行业冲击推演
- 企业做裁员、扩产、涨价、品牌危机预演
- 政策团队做刺激、封锁、监管变化的压力测试
- 游戏和虚拟世界做真正“像人”的 NPC 社会
- AI 公司构建更接近现实的训练环境和世界模型

## 验证快照

| 指标 | 当前仓库中的证据 |
| --- | --- |
| 真实经济体场景规模 | `817` 个 agent |
| 具名 LLM 决策者 | `32` 个高管和政府官员 |
| 家庭 LLM 代理 | 全部 `785` 个工人/消费者均通过 LLM 做个人财务决策 |
| 组织关系链 | `2,736` 条（雇佣、供应链、监管） |
| 历史验证 | 方向准确率 `84%`，`51 / 61` 项通过 |
| COVID 行为相似度 | `91%` 综合评分（7个维度） |
| 2008 CCI 匹配 | 模拟 `-43.6%` vs 真实 `-36.8%`（`84%` 匹配） |
| VIX 匹配 | 模拟峰值 `58` vs 真实峰值 `80`（`73%` 匹配） |
| Agent 可区分性 | 盲测准确率 `75%`（随机猜测的 `6倍`） |
| 单次运行 LLM 决策 | `500+` 高管 + `200+` 家庭决策 |

### Agent 实际做什么决策

**高管**（通过 `llm_agency.py`）：
- `CUT_HOURS`（典型危机中 237 次决策）
- `ABSORB_LOSSES`（75次 — 领导者自己承担损失保护员工）
- `EMERGENCY_FUND`（64次 — 政府部署救济）
- `RAISE_PRICES`、`ORGANIZE_UNION`、`CONFRONT_MANAGEMENT` 等

**家庭个人**（通过 `household_agency.py`）：
- 零工人员：`SEEK_SECOND_JOB` — *"接了更多网约车订单，取消了Netflix"*
- 退休人员：`CUT_DISCRETIONARY` — *"不再外出就餐，取消了订阅"*
- 工厂工人：`SEEK_SECOND_JOB` 或 `CUT_DISCRETIONARY`
- 学生：`SEEK_SECOND_JOB` — 打工还学费贷款
- 部分人：`ASK_FAMILY_HELP`、`TAP_SAVINGS`、`DEFER_BILLS`

## 三大核心引擎

### 1. Heart Engine

每个 agent 都有持续更新的内部心理状态：

- 唤醒度
- 情绪效价
- 紧张度
- 冲动控制
- 能量
- 脆弱度

同时还会保留伤痕、依恋风格、应对方式、威胁透镜和主观解释。也就是说，同样面对“油价翻倍”或者“疫情暴发”，两个工人不会做出同样的决定，因为他们不是用同一个模板在响应世界。

### 2. Ripple Engine

这个系统最有价值的地方之一，是它不把决策当成孤立事件，而是当成因果链。

例如：

```text
CEO 冻结招聘
-> 团队信心下降
-> 加班和收入被压缩
-> 家庭消费减少
-> 本地商户收入下滑
-> 更多企业开始保守收缩
```

这类传导不是一句“宏观压力上升”就结束，而是通过具名个体、公司和组织关系被具体追踪下来。

### 3. LLM Agency（高管决策）

LLM 在这里不是旁白，而是真正的决策者。系统里的高杠杆角色包括：

- `NovaTech` 的 Jensen Huang — 危机中加码或收缩
- `Federal_Reserve` 的 Jerome Mitchell — 批准紧急流动性
- `Treasury_Dept` 的 Janet Morrison — 部署紧急资金
- `FirstBank` 的 Jamie Stone — 承担损失或要求政府援助
- `US_Congress` 的 Rep. Williams — 在财政鹰派立场与紧急需要间权衡
- `CostPlus` 的 Ron Vachris — 尽量维持价格，最终不得不涨价

他们的输出是结构化决策，会直接改变世界状态，再由 Ripple Engine 向外传播。

### 3b. 家庭代理（个人决策）

每个非高管的 agent 也会通过 LLM 做个人财务决策：

- `CUT_DISCRETIONARY`：取消Netflix，不再外出就餐，买更便宜的品牌
- `SEEK_SECOND_JOB`：打零工，加班，做周末仓储工
- `TAP_SAVINGS`：动用储蓄来支付账单
- `ASK_FAMILY_HELP`：向亲戚借钱
- `DEFER_BILLS`：拖欠房租，与房东协商
- `ORGANIZE_COMMUNITY`：互助、集体行动

每个决策都反映 agent 的个性。恐惧匮乏的退休人员会削减开支；靠拼搏生存的零工会接更多单。这些厨房餐桌上的决策本身就是经济。

### 3c. 经济断路器

基于真实研究（FRED/BLS/NBER 数据）校准的防崩溃机制：

- **失业保险**：自动替代约45%收入（NBER：每美元效果是减税的8倍）
- **必要行业保护**：60-65%的经济体是抗衰退的（政府2200万员工 + 医疗 + 教育）
- **政府紧急雇佣**：失业率超过6%时，每周期吸收3%的失业者
- **刺激支票**：失业率超过10%时的广泛债务减免
- **FDIC等效**：防止银行体系完全崩溃

## 真实经济体层

仓库里的真实经济体 builder 已经包含：

- `NovaTech`
- `ApexDevices`
- `CloudScale`
- `MegaMart`
- `RetailGiant`
- `CostPlus`
- `FirstBank`
- `PetroMax`
- `PharmaCore`
- `Federal_Reserve`
- `US_Congress`
- `Treasury_Dept`
- `CDC`

再往下是大量普通人群：

- 科技从业者
- 工厂工人
- 零售员工
- 医护人员
- 零工劳动者
- 学生
- 退休人群
- 小企业主

这意味着它不是一个“agent_001 到 agent_999”的抽象沙盘，而是一个有组织结构、有社会角色、有情绪动力、有二阶连锁效应的可扩展世界。

## 商业价值

### 金融与投资

把“加息 50 个基点”“油价暴涨”“银行挤兑”“供应链断裂”输入系统，不只是看一个方向判断，而是看冲击是怎样穿透企业、劳动者、消费者、监管者和市场预期的。

### 企业战略

这类系统可以被用来推演：

- 裁员 20% 会造成什么二阶后果
- 提价、扩产、压资本开支，哪种路径更稳
- 品牌危机如何从舆论扩散到销售、监管和供应商

### 政策模拟

如果做全面封锁、财政刺激、金融救助或者更强监管，谁会先受损，谁会先恢复，机构信任会不会上升，社会凝聚力会不会断裂，这些都可以在更接近真实行为的系统里先做测试。

### 游戏与虚拟世界

对游戏行业来说，这类技术的意义很直接：NPC 不再是脚本木偶，而是能被经济、情绪、关系和记忆共同驱动的“社会个体”。

## 仓库结构

```text
api_server.py                               FastAPI + SSE 后端
frontend/                                   React/Vite 前端与 3D world viewer
src/prompt_forest/                          自适应路由、评估、记忆、编排
examples/learned_brain/world_sim/           世界模拟、场景、LLM 决策、验证脚本
examples/artifacts/                         历史验证和评估产物
artifacts/                                  实验报告、benchmark、运行产物
docs/architecture.md                        Prompt Forest 层的架构笔记
```

## 快速开始

### Python 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install fastapi uvicorn openai
```

这是一个研究型仓库，某些模拟脚本可能还依赖你本地已有的额外 ML 环境。

### 启动 UI 和后端

```bash
./start.sh
```

后端：`http://localhost:8000`  
前端：`http://localhost:3000`

### 查看真实经济体 builder

```bash
PYTHONPATH=examples/learned_brain python - <<'PY'
from world_sim.scenarios_real_economy import build_real_economy

world, agent_meta, fabric = build_real_economy(seed=42)
print("agents:", len(world.agents))
print("org_links:", len(fabric.links))
PY
```

### 运行历史验证

```bash
PYTHONPATH=examples/learned_brain python examples/learned_brain/world_sim/eval/historical_validation.py
PYTHONPATH=examples/learned_brain python examples/learned_brain/world_sim/eval/real_data_validation.py
```

### 运行旧版大城市场景叙事模拟

```bash
export OPENAI_API_KEY="your_key_here"
PYTHONPATH=src python examples/learned_brain/world_sim/run_large_narrative.py
```

## 下一步

这个项目真正有想象力的地方，不是“已经做完了什么”，而是它接下来能长成什么：

- 扩展到 `100,000+` agents
- 接入更多真实公司、机构和国际参与者
- 接入实时外部数据源，例如 FRED 和市场数据
- 把 3D world viewer 做成运营级控制台
- 让 agent 具备上网、搜集信息、更新信念的能力
- 支持多重危机叠加，比如通胀 + 裁员 + 能源冲击 + 政策冲突

## 投资视角下的意义

如果这条路线成立，它不会只是一个“很酷的模拟器”。它会变成一种新的基础设施：

- 企业级战略推演引擎
- AI agent 的高真实性训练环境
- 下一代经济智能产品
- 游戏和虚拟世界的人类行为底座
- 重大政策落地前的预演系统

更长远地看，软件不只是回答“世界发生了什么”，而是能先模拟“如果这样做，世界会怎样变化”。

## License

MIT
