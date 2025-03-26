from pytorch_lightning import Trainer


def main():
    num_nodes = 1
    accelerator = "auto"
    devices = 2
    strategy = "ddp"

    trainer = Trainer(accelerator=accelerator, devices=devices, num_nodes=num_nodes)

    print(f"{trainer.global_rank = }")


if __name__ == "__main__":
    main()
