import json


def test_manifest_records_have_required_fields(toy_dataset_root):
    train_manifest = toy_dataset_root / "train.json"
    records = json.loads(train_manifest.read_text(encoding="utf-8"))

    assert isinstance(records, list)
    assert len(records) == 2

    for record in records:
        assert set(record) >= {"id", "pcd_path", "diagram_path", "tokens_path"}
        assert (toy_dataset_root / record["pcd_path"]).exists()
        assert (toy_dataset_root / record["diagram_path"]).exists()
        assert (toy_dataset_root / record["tokens_path"]).exists()


def test_manifest_ids_are_unique_within_split(toy_dataset_root):
    train_manifest = toy_dataset_root / "train.json"
    records = json.loads(train_manifest.read_text(encoding="utf-8"))

    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids))
