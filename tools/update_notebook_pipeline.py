import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


path = Path("Hybrid_Movie_Recommendation1_realtime_streamlit_sync.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))
prefix = nb["cells"][:52]
demo_start = next(
    i
    for i, cell in enumerate(nb["cells"])
    if "".join(cell.get("source", [])).strip().startswith("## 9. Interactive Demo")
)
suffix = nb["cells"][demo_start:]

new_cells = [
    md(
        """## 8. Improved Offline Training, Validation & Evaluation Pipeline

Phần này là pipeline để so sánh công bằng các mô hình theo bài toán **top-K recommendation**.

- Tách `train / validation / test` để tune trên validation và chỉ báo cáo kết quả cuối trên test.
- Dùng chung metric `Precision@K`, `Recall@K`, `NDCG@K`.
- Bổ sung baseline mạnh hơn ngoài popularity.
- Tập trung vào ranking models trước khi quyết định model production."""
    ),
    code(Path("tools/notebook_cells/53_setup.py").read_text(encoding="utf-8")),
    md("### 8.1. Metric top-K dùng chung"),
    code(Path("tools/notebook_cells/55_metrics.py").read_text(encoding="utf-8")),
    md("### 8.2. Baseline mạnh hơn"),
    code(Path("tools/notebook_cells/57_baselines.py").read_text(encoding="utf-8")),
    md("### 8.3. Matrix Factorization tối ưu theo ranking"),
    code(Path("tools/notebook_cells/59_mf.py").read_text(encoding="utf-8")),
    md("### 8.4. BPR-MF và ALS implicit"),
    code(Path("tools/notebook_cells/61_ranking_models.py").read_text(encoding="utf-8")),
    md("### 8.5. Tune content dimensions theo metric ranking"),
    code(Path("tools/notebook_cells/63_content_tuning.py").read_text(encoding="utf-8")),
    md("### 8.6. Hybrid có kiểm soát và learned blender"),
    code(Path("tools/notebook_cells/65_hybrid.py").read_text(encoding="utf-8")),
    md("### 8.7. Benchmark cuối cùng trên test set"),
    code(Path("tools/notebook_cells/67_benchmark.py").read_text(encoding="utf-8")),
    md("### 8.8. Kiểm tra temporal split"),
    code(Path("tools/notebook_cells/69_temporal.py").read_text(encoding="utf-8")),
    md("### 8.9. Retrain full-data và lưu artifacts production"),
    code(Path("tools/notebook_cells/71_artifacts.py").read_text(encoding="utf-8")),
]

nb["cells"] = prefix + new_cells + suffix
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook updated:", len(nb["cells"]), "cells")
