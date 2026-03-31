import argparse
import json

from dotenv import load_dotenv

from src.crew_multimodal import MultiAgentFolderVisionQA


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="多 agent 文件夹图像问答执行脚本")
    parser.add_argument("--input-folder", required=True, help="输入图片文件夹")
    parser.add_argument("--prompt", required=True, help="用户问题/提示词")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Agent LLM 模型(支持 GPT/Claude)")
    parser.add_argument("--vision-model", default=None, help="视觉模型(默认跟随 --model)")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "gpt", "claude"],
        help="视觉模型提供方(auto/gpt/claude)",
    )
    parser.add_argument(
        "--comm-file",
        default="./runtime/agent_bus.jsonl",
        help="agent 通信文件(JSONL)",
    )
    parser.add_argument(
        "--output",
        default="./runtime/result.json",
        help="输出结果 JSON 路径",
    )
    args = parser.parse_args()

    app = MultiAgentFolderVisionQA(
        model=args.model,
        vision_model=args.vision_model,
        provider=args.provider,
        comm_file=args.comm_file,
    )
    result = app.run(input_folder=args.input_folder, prompt=args.prompt)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
