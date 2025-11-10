"""Memory profiling utilities for debugging VRAM usage."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Get memory usage of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)


def format_memory(memory_mb: float) -> str:
    """Format memory size with appropriate unit."""
    if memory_mb < 1:
        return f"{memory_mb * 1024:.2f} KB"
    elif memory_mb < 1024:
        return f"{memory_mb:.2f} MB"
    else:
        return f"{memory_mb / 1024:.2f} GB"


def profile_model_memory(model: nn.Module, device: torch.device) -> Dict:
    """
    Profile memory usage of model parameters and buffers.
    
    Args:
        model: PyTorch model
        device: Device where model is located
        
    Returns:
        Dictionary containing memory breakdown
    """
    param_memory = 0
    buffer_memory = 0
    layer_memory = defaultdict(float)
    layer_params = defaultdict(int)
    
    # Profile parameters
    for name, param in model.named_parameters():
        mem = get_tensor_memory_mb(param)
        param_memory += mem
        
        # Get layer name (first part of the parameter name)
        layer_name = name.split('.')[0] if '.' in name else name
        layer_memory[layer_name] += mem
        layer_params[layer_name] += param.numel()
    
    # Profile buffers (batch norm running stats, etc.)
    for name, buffer in model.named_buffers():
        mem = get_tensor_memory_mb(buffer)
        buffer_memory += mem
        
        layer_name = name.split('.')[0] if '.' in name else name
        layer_memory[layer_name] += mem
    
    total_memory = param_memory + buffer_memory
    
    return {
        'param_memory_mb': param_memory,
        'buffer_memory_mb': buffer_memory,
        'total_memory_mb': total_memory,
        'layer_memory': dict(layer_memory),
        'layer_params': dict(layer_params)
    }


def estimate_optimizer_memory(model: nn.Module, optimizer_type: str = 'adam') -> float:
    """
    Estimate optimizer state memory.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'sgd', etc.)
        
    Returns:
        Estimated memory in MB
    """
    param_memory = sum(get_tensor_memory_mb(p) for p in model.parameters() if p.requires_grad)
    
    # Optimizer state multipliers
    # Adam: 2x (momentum + variance)
    # SGD with momentum: 1x
    # SGD without momentum: ~0x
    multipliers = {
        'adam': 2.0,
        'adamw': 2.0,
        'sgd': 1.0,  # assuming momentum
        'rmsprop': 1.0
    }
    
    multiplier = multipliers.get(optimizer_type.lower(), 2.0)
    return param_memory * multiplier


def estimate_gradient_memory(model: nn.Module) -> float:
    """
    Estimate memory for gradients (same size as parameters).
    
    Args:
        model: PyTorch model
        
    Returns:
        Estimated memory in MB
    """
    return sum(get_tensor_memory_mb(p) for p in model.parameters() if p.requires_grad)


def estimate_activation_memory(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: torch.device,
    sample: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate activation memory by running a forward pass.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        device: Device to run on
        sample: If True, run actual forward pass; if False, return estimate
        
    Returns:
        Tuple of (total_activation_memory_mb, layer_activations)
    """
    if not sample:
        # Rough estimation without forward pass
        # This is very approximate
        batch_size, channels, height, width = input_size
        estimated_mb = (batch_size * channels * height * width * 4) / (1024 ** 2)  # float32
        return estimated_mb * 10, {}  # Multiply by ~10 for intermediate layers
    
    model.eval()
    layer_activations = {}
    activation_memory = 0
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                mem = get_tensor_memory_mb(output)
                layer_activations[name] = mem
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run forward pass
    try:
        with torch.no_grad():
            dummy_input = torch.randn(input_size).to(device)
            _ = model(dummy_input)
            activation_memory = sum(layer_activations.values())
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
        # Clean up
        if 'dummy_input' in locals():
            del dummy_input
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    model.train()
    return activation_memory, layer_activations


def get_gpu_memory_stats(device: torch.device) -> Dict:
    """
    Get current GPU memory statistics.
    
    Args:
        device: CUDA device
        
    Returns:
        Dictionary with memory stats in MB
    """
    if device.type != 'cuda':
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'free_mb': 0,
            'total_mb': 0
        }
    
    # Get memory in bytes and convert to MB
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(device)
        total = device_props.total_memory / (1024 ** 2)
        free = total - allocated
    else:
        total = 0
        free = 0
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'free_mb': free,
        'total_mb': total
    }


def print_memory_report(
    model: nn.Module,
    device: torch.device,
    config: Dict,
    detailed: bool = True,
    estimate_activations: bool = True
):
    """
    Print comprehensive memory usage report.
    
    Args:
        model: PyTorch model
        device: Device where model is located
        config: Training configuration
        detailed: If True, show per-layer breakdown
        estimate_activations: If True, estimate activation memory (runs forward pass)
    """
    print("\n" + "=" * 70)
    print("MEMORY PROFILING REPORT")
    print("=" * 70)
    
    # 1. GPU Memory Stats (before profiling)
    if device.type == 'cuda':
        gpu_stats = get_gpu_memory_stats(device)
        print(f"\n GPU Memory (Device: {device}):")
        print(f"  Total VRAM:      {format_memory(gpu_stats['total_mb'])}")
        print(f"  Currently Used:  {format_memory(gpu_stats['allocated_mb'])}")
        print(f"  Reserved by PyTorch: {format_memory(gpu_stats['reserved_mb'])}")
        print(f"  Available:       {format_memory(gpu_stats['free_mb'])}")
    
    # 2. Model Parameters Memory
    print(f"\n Model Memory Breakdown:")
    model_mem = profile_model_memory(model, device)
    print(f"  Parameters:      {format_memory(model_mem['param_memory_mb'])}")
    print(f"  Buffers:         {format_memory(model_mem['buffer_memory_mb'])}")
    print(f"  Total Model:     {format_memory(model_mem['total_memory_mb'])}")
    
    # 3. Training Memory Estimates
    print(f"\n  Training Memory Estimates:")
    
    # Gradients
    grad_mem = estimate_gradient_memory(model)
    print(f"  Gradients:       {format_memory(grad_mem)}")
    
    # Optimizer
    optimizer_type = config.get('training', {}).get('optimizer', {}).get('type', 'adam')
    opt_mem = estimate_optimizer_memory(model, optimizer_type)
    print(f"  Optimizer ({optimizer_type.upper()}): {format_memory(opt_mem)}")
    
    # Activations
    if estimate_activations:
        try:
            dataset_config = config.get('dataset', {})
            image_size = dataset_config.get('image_size', [512, 512])
            in_channels = config['model'].get('in_channels', 3)
            batch_size = config['data'].get('batch_size', 1)
            input_size = (batch_size, in_channels, image_size[0], image_size[1])
            
            print(f"  Estimating activations (input: {input_size})...")
            act_mem, layer_acts = estimate_activation_memory(model, input_size, device, sample=True)
            print(f"  Activations:     {format_memory(act_mem)}")
        except Exception as e:
            print(f"  Activations:     Could not estimate ({str(e)})")
            act_mem = 0
    else:
        act_mem = 0
    
    # 4. Total Estimated Memory
    total_estimated = model_mem['total_memory_mb'] + grad_mem + opt_mem + act_mem
    print(f"\n Total Estimated Training Memory: {format_memory(total_estimated)}")
    
    if device.type == 'cuda':
        percentage = (total_estimated / gpu_stats['total_mb']) * 100
        print(f"   (~{percentage:.1f}% of available VRAM)")
    
    # 5. Detailed Layer Breakdown
    if detailed:
        print(f"\n Per-Layer Memory Breakdown (Top 15):")
        print(f"  {'Layer Name':<40} {'Parameters':<15} {'Memory':<12}")
        print(f"  {'-' * 40} {'-' * 15} {'-' * 12}")
        
        # Sort layers by memory usage
        sorted_layers = sorted(
            model_mem['layer_memory'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (layer_name, mem) in enumerate(sorted_layers[:15]):
            params = model_mem['layer_params'].get(layer_name, 0)
            print(f"  {layer_name:<40} {params:<15,} {format_memory(mem):<12}")
        
        if len(sorted_layers) > 15:
            remaining = sum(mem for _, mem in sorted_layers[15:])
            print(f"  {'... (others)':<40} {'':<15} {format_memory(remaining):<12}")
    
    # 6. GPU Memory After Profiling
    if device.type == 'cuda':
        torch.cuda.synchronize()
        gpu_stats_after = get_gpu_memory_stats(device)
        print(f"\n📊 GPU Memory After Loading Model:")
        print(f"  Currently Used:  {format_memory(gpu_stats_after['allocated_mb'])}")
        print(f"  Available:       {format_memory(gpu_stats_after['free_mb'])}")
        
        # Warning if running low on memory
        if gpu_stats_after['free_mb'] < total_estimated * 0.5:
            print(f"\n  WARNING: Available memory may be insufficient!")
            print(f"   Consider reducing batch size or model size.")
    
    print("\n" + "=" * 70)


def profile_training_step_memory(
    model: nn.Module,
    train_loader,
    device: torch.device,
    criterion,
    optimizer
):
    """
    Profile memory usage during an actual training step.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        device: Device
        criterion: Loss function
        optimizer: Optimizer
    """
    if device.type != 'cuda':
        print("Memory profiling during training step only available for CUDA devices.")
        return
    
    print("\n" + "=" * 70)
    print("TRAINING STEP MEMORY PROFILE")
    print("=" * 70)
    
    model.train()
    
    # Get one batch
    batch = next(iter(train_loader))
    images = batch['image'].to(device) if isinstance(batch['image'], torch.Tensor) else torch.from_numpy(batch['image']).to(device)
    masks = batch['mask'].to(device) if isinstance(batch['mask'], torch.Tensor) else torch.from_numpy(batch['mask']).to(device)
    
    # Memory at different stages
    torch.cuda.reset_peak_memory_stats(device)
    
    stats = {}
    
    # 1. After loading batch
    torch.cuda.synchronize()
    stats['after_batch_load'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 2. After forward pass
    optimizer.zero_grad()
    outputs = model(images)
    torch.cuda.synchronize()
    stats['after_forward'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 3. After loss computation
    loss = criterion(outputs, masks)
    torch.cuda.synchronize()
    stats['after_loss'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 4. After backward pass
    loss.backward()
    torch.cuda.synchronize()
    stats['after_backward'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # 5. Peak memory
    stats['peak'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Print results
    print(f"\n Memory Usage Through Training Step:")
    print(f"  After batch load:   {format_memory(stats['after_batch_load'])}")
    print(f"  After forward:      {format_memory(stats['after_forward'])} (+{format_memory(stats['after_forward'] - stats['after_batch_load'])})")
    print(f"  After loss:         {format_memory(stats['after_loss'])} (+{format_memory(stats['after_loss'] - stats['after_forward'])})")
    print(f"  After backward:     {format_memory(stats['after_backward'])} (+{format_memory(stats['after_backward'] - stats['after_loss'])})")
    print(f"  Peak memory:        {format_memory(stats['peak'])}")
    
    print("\n" + "=" * 70)
    
    # Clean up
    del images, masks, outputs, loss
    torch.cuda.empty_cache()

