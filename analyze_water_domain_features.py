import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import ToTensor

from net.model import PromptIR


DOMAIN_PATHS = {
    "UIEB": "/mnt/disk1new/wzl/Semi-UIR/data/test/UIEB/input",
    "RUIE": "/mnt/disk1new/wzl/Semi-UIR/data/test/RUIE/input",
    "FUVD": "/mnt/disk1new/wzl/Semi-UIR/data/test/FUVD/input",
    "LSUI": "/mnt/disk1new/wzl/Semi-UIR/data/test/LSUI/input",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default="train_ckpt/uieb_water_promptir_e60/uieb-water-promptir-epoch=059.ckpt",
    )
    parser.add_argument("--domains", nargs="+", default=["UIEB", "RUIE", "FUVD", "LSUI"])
    parser.add_argument("--sample_per_domain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--output_dir", default="analysis/water_domain_features")
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


def list_images(domain_names, sample_per_domain, seed):
    rng = random.Random(seed)
    sampled = []
    for domain in domain_names:
        root = Path(DOMAIN_PATHS[domain])
        paths = sorted(
            p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )
        if len(paths) < sample_per_domain:
            raise RuntimeError(f"{domain} has only {len(paths)} images, need {sample_per_domain}")
        sampled_paths = rng.sample(paths, sample_per_domain)
        sampled.extend((domain, path) for path in sampled_paths)
    return sampled


def extract_features(net, items, device):
    to_tensor = ToTensor()
    captured = {"water": None}

    def water_hook(_, __, output):
        captured["water"] = output.detach()

    h1 = net.water_encoder.register_forward_hook(water_hook)

    degradation_features = []
    water_features = []
    labels = []
    paths = []

    with torch.no_grad():
        for idx, (domain, path) in enumerate(items, start=1):
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
            labels.append(domain)
            paths.append(str(path))
            if idx % 50 == 0:
                print(f"Extracted {idx}/{len(items)}")

    h1.remove()

    return (
        np.stack(degradation_features, axis=0),
        np.stack(water_features, axis=0),
        np.array(labels),
        paths,
    )


def fisher_discriminant_ratio(features, labels):
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    overall_mean = features.mean(axis=0, keepdims=True)
    between = 0.0
    within = 0.0
    for cls in classes:
        cls_feats = features[labels == cls]
        cls_mean = cls_feats.mean(axis=0, keepdims=True)
        between += len(cls_feats) * np.sum((cls_mean - overall_mean) ** 2)
        within += np.sum((cls_feats - cls_mean) ** 2)
    return float(between / (within + 1e-12))


def compute_metrics(features, labels):
    labels = np.asarray(labels)
    accs = []
    for seed in range(5):
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.3,
            random_state=seed,
            stratify=labels,
        )
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="auto"),
        )
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        accs.append(accuracy_score(y_test, preds))

    scaled = StandardScaler().fit_transform(features)
    return {
        "logreg_acc_mean": float(np.mean(accs)),
        "logreg_acc_std": float(np.std(accs)),
        "silhouette": float(silhouette_score(scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        "fisher_ratio": fisher_discriminant_ratio(scaled, labels),
    }


def run_tsne(features, labels, save_path, title):
    scaled = StandardScaler().fit_transform(features)
    perplexity = min(30, max(5, len(features) // 8))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(scaled)

    colors = {
        "UIEB": "#1f77b4",
        "RUIE": "#d62728",
        "FUVD": "#2ca02c",
        "LSUI": "#9467bd",
    }

    plt.figure(figsize=(7.2, 5.8), dpi=200)
    for domain in sorted(set(labels.tolist())):
        mask = labels == domain
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.85,
            label=domain,
            c=colors.get(domain, None),
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
                "logreg_acc_mean",
                "logreg_acc_std",
                "silhouette",
                "davies_bouldin",
                "fisher_ratio",
            ]
        )
        for name, metrics in metrics_by_name.items():
            writer.writerow(
                [
                    name,
                    f"{metrics['logreg_acc_mean']:.6f}",
                    f"{metrics['logreg_acc_std']:.6f}",
                    f"{metrics['silhouette']:.6f}",
                    f"{metrics['davies_bouldin']:.6f}",
                    f"{metrics['fisher_ratio']:.6f}",
                ]
            )


def save_summary(save_path, metrics_by_name):
    deg = metrics_by_name["degradation_related"]
    wat = metrics_by_name["water_domain_related"]
    lines = [
        "Condition Feature,Domain Separability",
        f"Degradation-related features,{deg['logreg_acc_mean']:.4f}",
        f"Water-domain-related features,{wat['logreg_acc_mean']:.4f}",
        "",
        "Interpretation:",
        "Degradation-related features,lower",
        "Water-domain-related features,higher",
    ]
    Path(save_path).write_text("\n".join(lines), encoding="utf-8")


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

    sampled = list_images(args.domains, args.sample_per_domain, args.seed)
    print(f"Using {len(sampled)} images across {len(args.domains)} domains")
    for domain in args.domains:
        print(f"{domain}: {args.sample_per_domain}")

    net = load_model(args.ckpt_path, device)
    degradation_features, water_features, labels, paths = extract_features(net, sampled, device)

    np.save(output_dir / "degradation_features.npy", degradation_features)
    np.save(output_dir / "water_domain_features.npy", water_features)
    np.save(output_dir / "labels.npy", labels)
    (output_dir / "sample_paths.txt").write_text("\n".join(paths), encoding="utf-8")

    degradation_metrics = compute_metrics(degradation_features, labels)
    water_metrics = compute_metrics(water_features, labels)
    metrics_by_name = {
        "degradation_related": degradation_metrics,
        "water_domain_related": water_metrics,
    }

    run_tsne(
        degradation_features,
        labels,
        output_dir / "tsne_degradation_features.png",
        "t-SNE of degradation features",
    )
    run_tsne(
        water_features,
        labels,
        output_dir / "tsne_water_domain_features.png",
        "t-SNE of water-domain features",
    )

    save_metrics_table(output_dir / "domain_separability_metrics.csv", metrics_by_name)
    save_summary(output_dir / "domain_separability_summary.csv", metrics_by_name)

    print("Degradation-related metrics:")
    for key, value in degradation_metrics.items():
        print(f"  {key}: {value:.6f}")
    print("Water-domain-related metrics:")
    for key, value in water_metrics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Saved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
