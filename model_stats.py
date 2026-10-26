# -*- coding: utf-8 -*-
"""
Calculate the number of trainable parameters and inference speed of the Stereo Matching model.
Supports separate timing for neural network and ODE solver.
"""

import time
import torch
import numpy as np
from omegaconf import OmegaConf

from core.endo_stereo_fm import EndoStereoFM


def count_trainable_params(model):
    """Count the number of trainable parameters in millions."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / 1e6  # Convert to millions


def measure_inference_speed(model, device, input_size, iters, time_nn_only=True):
    """
    Measure the inference speed of the model.
    
    Args:
        model: The model to measure
        device: Device to run inference on
        input_size: Input image size (H, W)
        iters: Number of iterations for the model
        time_nn_only: If True, skip timing of ODE solver by using Euler method
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    B, C, H, W = 1, 3, input_size[0], input_size[1]
    left = torch.randn(B, C, H, W).to(device)
    right = torch.randn(B, C, H, W).to(device)
    
    # Warmup
    print("Warming up for 5 iterations...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(left, right, iters=iters, test_mode=True)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing
    num_runs = 50
    nn_times = []
    
    print("Measuring inference speed over {} runs...".format(num_runs))
    
    if time_nn_only:
        # Patch model to use fast Euler method instead of ODE solver
        print("Using fast Euler method (excluding slow ODE solver)...")
        original_use_ode = model.flow_sampler_field.use_ode
        model.flow_sampler_field.use_ode = False
    
    for i in range(num_runs):
        with torch.no_grad():
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            disp_pred = model(left, right, iters=iters, test_mode=True)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            nn_times.append((end - start) * 1000)  # Convert to ms
    
    if time_nn_only:
        # Restore ODE setting
        model.flow_sampler_field.use_ode = original_use_ode
    
    nn_times = np.array(nn_times)
    
    return {
        'nn_mean_ms': nn_times.mean(),
        'nn_std_ms': nn_times.std(),
        'nn_min_ms': nn_times.min(),
        'nn_max_ms': nn_times.max(),
        'input_size': input_size,
        'iters': iters,
        'device': device
    }


def measure_nn_and_ode_speed(model, device, input_size, iters):
    """
    Measure both neural network and ODE solver time separately.
    """
    model.eval()
    model.to(device)
    
    # Create dummy input
    B, C, H, W = 1, 3, input_size[0], input_size[1]
    left = torch.randn(B, C, H, W).to(device)
    right = torch.randn(B, C, H, W).to(device)
    
    # Warmup
    print("Warming up for 5 iterations...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(left, right, iters=iters, test_mode=True)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing with separate measurements
    num_runs = 50
    nn_times_no_ode = []
    nn_times_with_ode = []
    
    print("Measuring neural network vs ODE solver time over {} runs...".format(num_runs))
    
    for i in range(num_runs):
        with torch.no_grad():
            # Measure NN only (Euler method, very fast)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            original_use_ode = model.flow_sampler_field.use_ode
            model.flow_sampler_field.use_ode = False
            
            start_nn = time.perf_counter()
            _ = model(left, right, iters=iters, test_mode=True)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_nn = time.perf_counter()
            nn_times_no_ode.append((end_nn - start_nn) * 1000)
            
            # Restore ODE setting
            model.flow_sampler_field.use_ode = original_use_ode
            
            # Measure full time (with ODE solver)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_full = time.perf_counter()
            _ = model(left, right, iters=iters, test_mode=True)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_full = time.perf_counter()
            nn_times_with_ode.append((end_full - start_full) * 1000)
    
    nn_times_no_ode = np.array(nn_times_no_ode)
    nn_times_with_ode = np.array(nn_times_with_ode)
    ode_times = nn_times_with_ode - nn_times_no_ode
    
    return {
        'nn_mean_ms': nn_times_no_ode.mean(),
        'nn_std_ms': nn_times_no_ode.std(),
        'ode_mean_ms': ode_times.mean(),
        'ode_std_ms': ode_times.std(),
        'total_mean_ms': nn_times_with_ode.mean(),
        'total_std_ms': nn_times_with_ode.std(),
        'input_size': input_size,
        'iters': iters,
        'device': device
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate model parameters and inference speed')
    parser.add_argument('--config_path', type=str, default='./config/train_scared.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--input_h', type=int, default=320,
                        help='Input image height')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input image width')
    parser.add_argument('--iters', type=int, default=32,
                        help='Number of model iterations')
    parser.add_argument('--separate_nn_ode', action='store_true',
                        help='Separate NN and ODE timing')
    
    args = parser.parse_args()
    
    print("Loading model...")
    cfg = OmegaConf.load(args.config_path)
    model = EndoStereoFM(cfg)
    print("Model loaded successfully!")
    
    # Count parameters
    params_M = count_trainable_params(model)
    print("\n" + "="*60)
    print("Trainable Parameters: {:.2f} M".format(params_M))
    print("="*60)
    
    # Measure inference speed
    input_size = (args.input_h, args.input_w)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    if args.separate_nn_ode:
        # Separate NN and ODE timing
        timing = measure_nn_and_ode_speed(model, args.device, input_size, args.iters)
        
        print("\n" + "="*60)
        print("INFERENCE SPEED (Separated)")
        print("="*60)
        print("Input size:     {}".format(timing['input_size']))
        print("Iters:          {}".format(timing['iters']))
        print("Device:         {}".format(timing['device']))
        print("-"*60)
        print("Neural Network: {:.2f} +/- {:.2f} ms".format(timing['nn_mean_ms'], timing['nn_std_ms']))
        print("ODE Solver:     {:.2f} +/- {:.2f} ms".format(timing['ode_mean_ms'], timing['ode_std_ms']))
        print("Total:          {:.2f} +/- {:.2f} ms".format(timing['total_mean_ms'], timing['total_std_ms']))
        print("="*60)
    else:
        # NN only timing (using fast Euler method, excluding slow ODE solver)
        timing = measure_inference_speed(model, args.device, input_size, args.iters, time_nn_only=True)
        
        print("\n" + "="*60)
        print("Neural Network Speed (excluding ODE solver)")
        print("="*60)
        print("Input size:     {}".format(timing['input_size']))
        print("Iters:          {}".format(timing['iters']))
        print("Device:         {}".format(timing['device']))
        print("-"*60)
        print("Speed (ms):     {:.2f} +/- {:.2f}".format(timing['nn_mean_ms'], timing['nn_std_ms']))
        print("Min (ms):       {:.2f}".format(timing['nn_min_ms']))
        print("Max (ms):       {:.2f}".format(timing['nn_max_ms']))
        print("="*60)
