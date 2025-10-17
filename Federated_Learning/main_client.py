# main_client.py
import argparse
from flwr.client import start_client
from my_flower_app.client_fn import client_fn
from my_flower_app.client_app import DEVICE


def run_client(
    partition_id: int,
    dataset_path: str = "dataset_opsi_part",
    batch_size: int = 8,
    img_size=224,
    num_workers: int = 2,
    lr: float = 1e-4,
    grad_accum_steps: int = 1,
    local_epochs: int = 5,
):
    print(f"[INFO] Starting client {partition_id} on device {DEVICE}")

    client_dataset_path = f"{dataset_path}/partition_{partition_id}"

    fl_client = client_fn(
        dataset_path=client_dataset_path,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        local_epochs=local_epochs,
    )

    start_client(
        server_address="127.0.0.1:8090",
        client=fl_client,
        grpc_max_message_length=1024 * 1024 * 1024,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--partition-id", type=int, required=True, help="Client partition ID")
    parser.add_argument("--dataset-path", type=str, default="dataset_opsi_part", help="Root dataset folder")
    parser.add_argument("--batch-size", type=int, default=8, help="Local batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image size (square). Use 224 for safety")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per federated round")
    args = parser.parse_args()

    run_client(
        partition_id=args.partition_id,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        local_epochs=args.local_epochs,
    )
