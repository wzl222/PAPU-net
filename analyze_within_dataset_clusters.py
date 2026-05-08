import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor

from net.model import PromptIR


DATASET_PATHS = {
    "LSUI": "/mnt/disk1new/wzl/Semi-UIR/data/test/LSUI/input",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="LSUI", choices=sorted(DATASET_PATHS.keys()))
    parser.add_argument(
        "--ckpt_path",
        default="train_ckpt/uieb_water_promptir_e60/uieb-water-promptir-epoch=059.ckpt",
    )
    parser.add_argument("--sample_count", type=int, default=400)
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--output_dir", default="analysis/within_dataset_clusters_lsui")
    return parser.parse_args()


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    h_pad = (img_multiple_of - height % img_multiple_of) % img_multiple_of
    w_pad = (img_multiple_of - width % img_multiple_of) % img_multiple_of
    if h_pad or w_pad:
        input_ = F.pad(input_, (0, w_pad, 0, h_pad), "reflect")
    return input_


def load_model(ckpt_path, device):
    net = PromptIR(decoder=True, water_aware=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[4:] if key.startswith("net.") else key] = value
    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    net.to(device).eval()
    return net


def sample_images(dataset_name, sample_count, seed):
    root = Path(DATASET_PATHS[dataset_name])
    paths = sorted(
        p
        for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if len(paths) < sample_count:
        raise RuntimeError(f"{dataset_name} has only {len(paths)} images, need {sample_count}")
    rng = random.Random(seed)
    return rng.sample(paths, sample_count)


def extract_features(net, image_paths, device):
    to_tensor = ToTensor()
    captured = {"water": None}

    def water_hook(_, __, output):
        captured["water"] = output.detach()

    hook = net.water_encoder.register_forward_hook(water_hook)

    degradation_features = []
    water_features = []

    with torch.no_grad():
        for idx, path in enumerate(image_paths, start=1):
            image = np.array(Image.open(path).convert("RGB"))
            tensor = to_tensor(image).unsqueeze(0).to(device)
            tensor = pad_input(tensor)
            stats = net.water_encoder._stats(tensor).detach()
            _ = net(tensor)
            water = captured["water"]
            if water is None:
                raise RuntimeError(f"Feature hooks failed for {path}")
            degradation_features.append(stats.squeeze(0).cpu().numpy())
            water_features.append(water.squeeze(0).cpu().numpy())
            if idx % 50 == 0:
                print(f"Extracted {idx}/{len(image_paths)}")

    hook.remove()
    return np.stack(degradation_features, axis=0), np.stack(water_features, axis=0)


def cluster_and_score(features, num_clusters, seed):
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(scaled)
    metrics = {
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
    }
    return labels, metrics


def run_tsne(features, cluster_labels, save_path, title):
    scaled = StandardScaler().fit_transform(features)
    perplexity = min(30, max(5, len(features) // 10))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(scaled)

    cmap = plt.cm.get_cmap("tab10", len(np.unique(cluster_labels)))
    plt.figure(figsize=(7.2, 5.8), dpi=200)
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.85,
            label=f"Cluster {cluster_id + 1}",
            c=[cmap(cluster_id)],
        )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(title)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_metrics_table(save_path, metrics_by_name):
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature_type",
                "silhouette",
                "davies_bouldin",
                "calinski_harabasz",
            ]
        )
        for name, metrics in metrics_by_name.items():
            writer.writerow(
                [
                    name,
                    f"{metrics['silhouette']:.6f}",
                    f"{metrics['davies_bouldin']:.6f}",
                    f"{metrics['calinski_harabasz']:.6f}",
                ]
            )


def save_assignments(save_path, image_paths, deg_clusters, water_clusters):
    with open(save_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "degradation_cluster", "water_domain_cluster"])
        for path, deg_cluster, water_cluster in zip(image_paths, deg_clusters, water_clusters):
            writer.writerow([str(path), int(deg_cluster), int(water_cluster)])


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)

    image_paths = sample_images(args.dataset, args.sample_count, args.seed)
    print(f"Dataset: {args.dataset}")
    print(f"Sampled images: {len(image_paths)}")
    print(f"Clusters: {args.num_clusters}")

    net = load_model(args.ckpt_path, device)
    degradation_features, water_features = extract_features(net, image_paths, device)

    np.save(output_dir / "degradation_features.npy", degradation_features)
    np.save(output_dir / "water_domain_features.npy", water_features)
    (output_dir / "sample_paths.txt").write_text(
        "\n".join(str(path) for path in image_paths), encoding="utf-8"
    )

    deg_clusters, deg_metrics = cluster_and_score(
        degradation_features, args.num_clusters, args.seed
    )
    water_clusters, water_metrics = cluster_and_score(
        water_features, args.num_clusters, args.seed
    )

    run_tsne(
        degradation_features,
        deg_clusters,
        output_dir / "tsne_degradation_features.png",
        "t-SNE of degradation features",
    )
    run_tsne(
        water_features,
        water_clusters,
        output_dir / "tsne_water_domain_features.png",
        "t-SNE of water-domain features",
    )

    metrics_by_name = {
        "degradation_related": deg_metrics,
        "water_domain_related": water_metrics,
    }
    save_metrics_table(output_dir / "within_dataset_cluster_metrics.csv", metrics_by_name)
    save_assignments(
        output_dir / "cluster_assignments.csv", image_paths, deg_clusters, water_clusters
    )

    print("Degradation-related metrics:")
    for key, value in deg_metrics.items():
        print(f"  {key}: {value:.6f}")
    print("Water-domain-related metrics:")
    for key, value in water_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
