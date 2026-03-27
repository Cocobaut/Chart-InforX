# Demo Guide - `data_extraction_v8.py`

## 1) Muc tieu file

`data_extraction_v8.py` la pipeline trich xuat gia tri tu bieu do cot:

1. YOLO detect objects (plot, bar, legend)
2. detect axes
3. extract text labels
4. compute ratio
5. match legend patches
6. extract embeddings
7. match legend to bars
8. compute bar values
9. save output (Excel + JSON)

## 2) Cac ham debug va tac dung

### `_normalize_debug_options(debug_options)`
- Tron config debug do nguoi dung truyen vao voi gia tri mac dinh.
- Dam bao cac khoa debug luon ton tai.

### `_debug_log(debug_options, message)`
- In log text khi `enabled=True` va `print_logs=True`.
- Dung de hien thi thong tin trace trong pipeline.

### `_debug_show_patch(debug_options, patch, kind, bbox_xyxy)`
- Hien thi patch (legend/bar) bang matplotlib.
- Chi chay khi `enabled=True` va `show_patch=True`.
- Duoc goi trong `extract_patch_embedding(...)`.

### `_debug_print_bar_similarity(...)`
- In score similarity giua moi bar va legend duoc chon.
- Chi chay khi `enabled=True` va `show_similarity=True`.
- Huu ich khi can kiem tra logic match mau.

### `_debug_save_overlay(...)`
- Ve va luu anh debug tong hop (axes, legend, bars, mapping lines).
- Chi chay khi `enabled=True`, `save_overlay=True`, va co `debug_dir`.
- Goi ham `draw_debug_image(...)` de tao file anh debug.

## 3) Cac co debug co the bat/tat

`debug_options` (truyen vao `getYVal(...)`):

- `enabled`: bat/tat toan bo debug
- `print_logs`: in log text
- `show_patch`: hien thi patch legend/bar
- `show_similarity`: in score similarity legend-bar
- `save_overlay`: luu anh debug overlay
- `debug_dir`: thu muc luu overlay

Mau cau hinh:

```python
debug_options = {
    "enabled": True,
    "print_logs": True,
    "show_patch": False,
    "show_similarity": True,
    "save_overlay": True,
    "debug_dir": str(OUT_DIR / "debug_viz"),
}
```

## 4) Huong dan chay demo tren Google Colab

### Buoc 1 - Chuan bi du lieu tren Google Drive

Can co cau truc toi thieu:

```text
<BASE_DIR>/
  images/
    *.png | *.jpg | *.jpeg
  json/
    <image_stem>.json
<WEIGHT_PATH>  (YOLO best.pt)
```

Luu y:
- Moi anh trong `images/` can co file JSON cung `stem` trong `json/`.
- JSON can co thong tin `task2/task3` de `getProbableLabels(...)` doc duoc.

### Buoc 2 - Mo Colab va bat GPU

1. Runtime -> Change runtime type -> Hardware accelerator = GPU
2. Mount Google Drive (file da co san lenh mount)

### Buoc 3 - Mo file code va sua config

Trong file co block config o gan cuoi:

- `USE_COLAB`
- `BASE_DIR`
- `WEIGHT_PATH`
- `OUT_DIR`
- `IMG_DIR`, `JSON_DIR`, `EXCEL_PATH`, `JSON_PATH`

Dam bao cac path nay dung voi Drive cua ban.

### Buoc 4 - (Tuy chon) Bat debug

Bo comment block `debug_options` va truyen vao:

```python
yValueDict = getYVal(
    IMG_DIR,
    JSON_DIR,
    objects_dectector,
    backbone,
    debug_options=debug_options
)
```

Neu khong can debug thi giu:

```python
yValueDict = getYVal(IMG_DIR, JSON_DIR, objects_dectector, backbone)
```

### Buoc 5 - Chay pipeline

Khi chay xong, ket qua duoc luu:

- `EXCEL_PATH` -> `y_values.xlsx`
- `JSON_PATH` -> `y_values.json`

Neu bat `save_overlay`, anh debug se nam trong `debug_dir`.

## 5) Goi y test nhanh

1. Chay voi 2-3 anh mau truoc (khong bat `show_patch`) de kiem tra toc do.
2. Bat `show_similarity=True` neu thay match legend-bar sai.
3. Bat `save_overlay=True` de nhin truc quan mapping bar/legend/x-label.

