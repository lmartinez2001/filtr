import torch
from fvcore.nn import FlopCountAnalysis

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

    print(f"--- Model Complexity Audit ---")
    print(f"Input Shape:      {tuple(tokens.shape)}")
    print(f"Total MACs:       {format_metric(mac_count, 'MACs')}")
    print(f"Total FLOPs:      {format_metric(total_flops, 'FLOPs')} (Standard: 2*MACs)")
    print(f"Params (Total):   {format_metric(params_total, 'Pts')}")
    print(f"Params (Train):   {format_metric(params_trainable, 'Pts')}")
    print("-" * 30)
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

    print(f"--- Model Complexity Audit (End-to-End) ---")
    print(f"Input Shape:      {tuple(inputs.shape)}")
    print(f"Total MACs:       {format_metric(mac_count, 'MACs')}")
    print(f"Total FLOPs:      {format_metric(total_flops, 'FLOPs')} (Standard: 2*MACs)")
    print(f"Params (Total):   {format_metric(params_total, 'Pts')}")
    print(f"Params (Train):   {format_metric(params_trainable, 'Pts')}")
    print("-" * 30)
    return total_flops, params_trainable
