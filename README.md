# CrewAI 多智能体（文件夹 + Prompt + 图片）示例

这个版本支持 **GPT 和 Claude** 两种模型族：
- Agent 编排模型：`--model`
- 视觉分析模型：`--vision-model`（默认跟随 `--model`）
- 视觉提供方：`--provider auto|gpt|claude`

流程拆分：

`user -> orchestrator -> workers -> synthesizer -> evaluator -> finalizer`

输入：
- `input_folder`：图片文件夹
- `prompt`：问题/任务描述

输出：
- 最终答案（含置信度）
- 使用到的图片路径
- 证据与限制项
- 有序通信文件（JSONL）

## Agent 拆分

1. **Orchestrator**：任务分解、选图、预算与停止条件
2. **VisionWorker**：图像事实提取（通过本地视觉工具）
3. **PromptWorker**：问题语义和判定标准抽取
4. **Synthesizer**：汇总 worker 输出形成草稿
5. **Evaluator**：质量审查和修正建议
6. **Finalizer**：生成用户可读最终答复

## 文件通信与上下文压缩

- 所有阶段都写入同一个有序 JSONL 文件（`seq` 递增）。
- 当接近上下文阈值时，会优先压缩可压缩阶段（orchestrator/workers/synthesis），尽量保留 evaluator/finalizer。
- 压缩后会写入 `context_summary`，防止因上下文过长直接失败。

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# GPT 需要
export OPENAI_API_KEY=your_openai_key
# Claude 需要
export ANTHROPIC_API_KEY=your_anthropic_key
```

### GPT 示例

```bash
python run_folder.py \
  --input-folder ./images \
  --prompt "请找出异常操作并说明风险" \
  --model gpt-4.1-mini \
  --provider gpt \
  --output ./runtime/result_gpt.json
```

### Claude 示例

```bash
python run_folder.py \
  --input-folder ./images \
  --prompt "请找出异常操作并说明风险" \
  --model claude-3-7-sonnet-latest \
  --provider claude \
  --output ./runtime/result_claude.json
```

## 本地工具

- `LocalImageCatalogTool`：扫描本地目录图片
- `VisionQATool`：读取本地图片并调用多模态模型（OpenAI/Anthropic）

