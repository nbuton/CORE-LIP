import argparse
from core_lip.engine.trainer import CORE_LIP_Trainer, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = get_config(args.config)
    cfg.config_path = args.config  # Store path for the trainer

    trainer = CORE_LIP_Trainer(cfg, device=args.device)
    trainer.run()
    trainer.plot()


if __name__ == "__main__":
    main()
