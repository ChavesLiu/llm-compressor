import os
import argparse
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot

# 当前脚本目录
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(model_path: str):
    """
    加载模型和分词器
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    return model, tokenizer


def load_dataset_auto(dataset_path: str):
    """
    加载数据集（支持 Huggingface ID 或 本地 JSON）
    """
    if os.path.exists(dataset_path):
        return load_dataset('json', data_files=dataset_path)['train']
    else:
        return load_dataset(dataset_path, split='train')


def preprocess_dataset(dataset, tokenizer):
    """
    将 message/text 转换为模型格式输入文本
    """
    def preprocess(example):
        messages = example.get("messages") or [{"role": "user", "content": example.get("text", "")}]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        }

    return dataset.map(preprocess)


def tokenize_dataset(dataset, tokenizer, max_length):
    """
    对数据进行分词
    """
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main(args):
    # 读取配置文件
    config_path = os.path.join(CURRENT_DIR, args.config_file)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    print(f"加载模型: {config['model']}")
    model, tokenizer = load_model(config["model"])

    print(f"加载校准数据集: {config['dataset_id']}")
    dataset = load_dataset_auto(config["dataset_id"])

    print("数据预处理...")
    dataset = preprocess_dataset(dataset, tokenizer)

    print("分词 Tokenize...")
    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        config.get("max_sequence_length", 512)
    )

    recipe_path = config["recipe"]
    if not os.path.isabs(recipe_path):
        recipe_path = os.path.join(CURRENT_DIR, recipe_path)

    print(f"启动量化，使用 recipe: {recipe_path}")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe_path,
        max_seq_length=config.get("max_sequence_length", 512),
        num_calibration_samples=config.get("num_calibration_samples", 200),
        save_compressed=True,
        trust_remote_code_model=True,
        output_dir=config["save_dir"],
    )

    print(f"量化完成，压缩模型已保存至: {config['save_dir']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm quantization')
    parser.add_argument('--config_file', required=True, help='YAML 配置文件路径')
    args = parser.parse_args()
    main(args)
