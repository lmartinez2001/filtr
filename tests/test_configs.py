from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def test_main_config_references_existing_default_files():
    config_path = Path("conf/config.yaml")
    cfg = OmegaConf.load(config_path)

    for default in cfg.defaults:
        if default == "_self_":
            continue

        group, name = next(iter(default.items()))
        referenced_path = Path("conf") / group / f"{name}.yaml"
        assert referenced_path.exists(), f"Missing config file: {referenced_path}"


def test_all_yaml_configs_parse():
    for config_path in Path("conf").rglob("*.yaml"):
        cfg = OmegaConf.load(config_path)
        assert cfg is not None, f"Failed to parse {config_path}"


def test_filtr_network_config_instantiates():
    cfg = OmegaConf.load("conf/model/filtr.yaml")
    model = instantiate(cfg.network)

    assert model.num_queries == cfg.network.num_queries
    assert model.decoder.d_model == cfg.network.decoder.d_model
