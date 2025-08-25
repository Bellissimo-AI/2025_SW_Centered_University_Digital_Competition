from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from safetensors.torch import load_file as load_safetensors

def get_model(args):
    """
    Load model and tokenizer given the configuration arguments.

    Args:
        args: Configuration object with attributes:
            - model_name
            - num_labels
            - ckpt_path
            - embedding_model

    Returns:
        Tuple (model, tokenizer)
    """
    print(f"Loading model: {args.model_name} (num_labels={args.num_labels})")

    if args.model_name == 'AvsHModel':
        from models.AvsHModel import AvsHModel

        model = AvsHModel(args)
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

        if args.ckpt_path:
            ckpt_file = f"{args.ckpt_path}/model.safetensors"
            print(f"Loading weights from safetensors checkpoint: {ckpt_file}")
            state_dict = load_safetensors(ckpt_file, device="cpu")
            model.load_state_dict(state_dict, strict=True)

    else:
        model_path = args.ckpt_path if args.ckpt_path else args.model_name
        if args.ckpt_path:
            print(f"Loading model from checkpoint: {args.ckpt_path}")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=args.num_labels,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                  trust_remote_code=True)

        # # 1) 우선 모든 파라미터를 얼리기
        # for param in model.parameters():
        #     param.requires_grad = False

        # # 2) classifier 헤드만 학습 가능하도록 해제
        # for name, param in model.named_parameters():
        #     if "classifier" in name:  
        #         param.requires_grad = True

        # # 3) 정상 설정됐는지 확인
        # print("---- trainable parameters ----")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)


    return model, tokenizer
