import logging
import torch
from fvcore.nn import FlopCountAnalysis

logger = logging.getLogger(__name__)

def get_model_complexity_info(model, tokens, pos):
    model.eval()

    flops_analyzer = FlopCountAnalysis(model, (tokens,pos))
    
    mac_count = flops_analyzer.total()
    
    total_flops = mac_count * 2

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_metric(value, unit):
        if value > 1e9:
            return f"{value / 1e9:.2f} G{unit}"
        elif value > 1e6:
            return f"{value / 1e6:.2f} M{unit}"
        elif value > 1e3:
            return f"{value / 1e3:.2f} K{unit}"
        return f"{value} {unit}"

    logger.info("--- Model Complexity Audit ---")
    logger.info("Input Shape:      %s", tuple(tokens.shape))
    logger.info("Total MACs:       %s", format_metric(mac_count, 'MACs'))
    logger.info("Total FLOPs:      %s (Standard: 2*MACs)", format_metric(total_flops, 'FLOPs'))
    logger.info("Params (Total):   %s", format_metric(params_total, 'Pts'))
    logger.info("Params (Train):   %s", format_metric(params_trainable, 'Pts'))
    logger.info("%s", "-" * 30)
    return total_flops, params_trainable


def get_e2e_model_complexity_info(model, inputs):
    model.eval()

    flops_analyzer = FlopCountAnalysis(model, inputs)
    
    mac_count = flops_analyzer.total()
    
    total_flops = mac_count * 2

    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def format_metric(value, unit):
        if value > 1e9:
            return f"{value / 1e9:.2f} G{unit}"
        elif value > 1e6:
            return f"{value / 1e6:.2f} M{unit}"
        elif value > 1e3:
            return f"{value / 1e3:.2f} K{unit}"
        return f"{value} {unit}"

    logger.info("--- Model Complexity Audit (End-to-End) ---")
    logger.info("Input Shape:      %s", tuple(inputs.shape))
    logger.info("Total MACs:       %s", format_metric(mac_count, 'MACs'))
    logger.info("Total FLOPs:      %s (Standard: 2*MACs)", format_metric(total_flops, 'FLOPs'))
    logger.info("Params (Total):   %s", format_metric(params_total, 'Pts'))
    logger.info("Params (Train):   %s", format_metric(params_trainable, 'Pts'))
    logger.info("%s", "-" * 30)
    return total_flops, params_trainable
