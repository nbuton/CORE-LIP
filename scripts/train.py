import argparse
from core_lip.engine.trainer import CORE_LIP_Trainer, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = get_config(args.config)

    trainer = CORE_LIP_Trainer(cfg, args.config, device=args.device)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    best_auc = trainer.run()
    print("best_auc:", best_auc)
    trainer.plot()


if __name__ == "__main__":
    main()
