# Written by: Inal A. Mashukov
# MIT License.



import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from transformer import * 

def trace_oracle_dimensions():
    """
    Step-by-step tensor shape tracker to verify the structural integrity
    of the ORACLE architecture on the GPU.
    """
    print("="*50)
    print("ORACLE ARCHITECTURE & TENSOR TRACER")
    print("="*50)
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- USING DEVICE: {device.type.upper()} ---")

    # 1. Define hyperparameters
    B, L, C = 2, 10, 2
    d_model = 16 
    nheads = 2
    patch_size = 2
    stride = 1
    num_layers = 1
    forecast_len = 4

    # Initialize the model and move to device
    model = ORACLE(
        lookback=L, patch_size=patch_size, stride=stride, d_model=d_model,
        nheads=nheads, num_layers=num_layers, forecast_len=forecast_len
    ).to(device)
    
    print("\n--- FULL ARCHITECTURE ---")
    print(model)
    print("-" * 25)

    # 2. Create the fake B=2, L=10, C=2 tensor and move to device
    print("\n--- FORWARD PASS TRACE ---")
    x = torch.randn(B, L, C, device=device)
    print(f"1. Input 'x' shape: \n   {x.shape} -> (Batch, Lookback, Channels) on {x.device}")

    # 3. Temporal Stream
    z_t, patches = model.temporal_embed(x)
    print(f"\n2. Temporal Embedding output 'z_t' shape: \n   {z_t.shape} -> (B*C, Num_Patches, d_model) on {z_t.device}")
    print(f"   Raw unfold 'patches' shape: \n   {patches.shape} -> (B, C, Num_Patches, Patch_Size)")

    # 4. Feature Stream
    z_f = model.feature_embed(x)
    print(f"\n3. Feature Embedding output 'z_f' shape: \n   {z_f.shape} -> (Batch, Channels, d_model)")

    # 5. Co-Attention Block
    N = patches.shape[-1]
    for i, layer in enumerate(model.layers):
        z_t, z_f = layer(z_t, z_f, B, C, N, use_time=True, use_feature=True)
        print(f"\n4. Co-Attention Layer {i} outputs:")
        print(f"   'z_t' shape: {z_t.shape}")
        print(f"   'z_f' shape: {z_f.shape}")

    # 6. Final Norms
    z_t = model.norm_t_final(z_t)
    z_f = model.norm_f_final(z_f)
    print(f"\n5. Post-Norm shapes remain identical.")

    # 7. Output Routing / Broadcasting
    z_t_flat = z_t.reshape(B, C, -1)
    print(f"\n6. Flattened Temporal stream 'z_t_flat' shape: \n   {z_t_flat.shape} -> (Batch, Channels, Num_Patches * d_model)")
    
    z_combined = torch.cat([z_t_flat, z_f], dim=-1)
    print(f"\n7. Joint Concatenation 'z_combined' shape: \n   {z_combined.shape} -> (Batch, Channels, Joint_Dim)")

    # 8. Forecasting Head
    out = model.head(z_combined)
    print(f"\n8. Final Output shape: \n   {out.shape} -> (Batch, Channels, Forecast_Len) on {out.device}")
    print("="*50)


def test_oracle_learning():
    """
    A sanity-check training loop to ensure gradients flow properly and 
    the architecture can learn a simple deterministic function, GPU-accelerated.
    """
    print("\n Starting Toy Training Loop (Sine/Cosine Forecasting)...")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- TRAINING ON: {device.type.upper()} ---")
    
    # Hyperparameters
    B, L, C = 2, 10, 2
    forecast_len = 4
    
    # Create the model and move to device
    model = ORACLE(
        lookback=L, patch_size=2, stride=1, d_model=16,
        nheads=2, num_layers=1, forecast_len=forecast_len
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Create a deterministic dataset: sin(x) for channel 0, cos(x) for channel 1
    t_input = torch.linspace(0, 4, L).unsqueeze(0).repeat(B, 1) # (B, L)
    t_target = torch.linspace(4.5, 6, forecast_len).unsqueeze(0).repeat(B, 1) # (B, H)

    # Build input tensor X (B, L, C) and target tensor Y (B, C, H)
    x = torch.zeros(B, L, C)
    x[:, :, 0] = torch.sin(t_input)
    x[:, :, 1] = torch.cos(t_input)
    
    y = torch.zeros(B, C, forecast_len)
    y[:, 0, :] = torch.sin(t_target)
    y[:, 1, :] = torch.cos(t_target)

    # Move tensors to device
    x = x.to(device)
    y = y.to(device)

    # Training Loop
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(x) 
        
        # Loss and Backprop
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/500], MSE Loss: {loss.item():.6f}")

    print("\nTraining complete. Generating and saving visualization...")
    model.eval()
    with torch.no_grad():
        final_preds = model(x)
        
        # Extract data for Batch 0 and move strictly back to CPU for Matplotlib
        t_in = t_input[0].cpu().numpy()
        t_out = t_target[0].cpu().numpy()
        
        x_in_f0 = x[0, :, 0].cpu().numpy()
        y_true_f0 = y[0, 0, :].cpu().numpy()
        y_pred_f0 = final_preds[0, 0, :].cpu().numpy()
        
        x_in_f1 = x[0, :, 1].cpu().numpy()
        y_true_f1 = y[0, 1, :].cpu().numpy()
        y_pred_f1 = final_preds[0, 1, :].cpu().numpy()

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Feature 0 (Sine)
        ax1.plot(t_in, x_in_f0, label='Input (Lookback)', marker='o', color='gray')
        ax1.plot(t_out, y_true_f0, label='True Target', marker='o', color='blue')
        ax1.plot(t_out, y_pred_f0, label='ORACLE Prediction', marker='x', linestyle='--', color='red', markersize=8)
        ax1.set_title("Feature 0: Sine Wave Forecasting")
        ax1.set_ylabel("Amplitude")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Feature 1 (Cosine)
        ax2.plot(t_in, x_in_f1, label='Input (Lookback)', marker='o', color='gray')
        ax2.plot(t_out, y_true_f1, label='True Target', marker='o', color='blue')
        ax2.plot(t_out, y_pred_f1, label='ORACLE Prediction', marker='x', linestyle='--', color='red', markersize=8)
        ax2.set_title("Feature 1: Cosine Wave Forecasting")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Amplitude")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        
        # SAVE THE PLOT TO DIRECTORY
        save_path = "model_forecast_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {save_path}")
        
        # Display the plot
        plt.show()



def test_patching():
    print("Running Patching Tests...\n")
    print("=" * 80)
    batch_size = 2
    channels = 7
    patch_size = 16
    stride = 16
    
    patcher = Patching(patch_size=patch_size, stride=stride, pad_end=True)

    print("=== TEST 1: Perfectly Fitting Sequence ===")
    lookback_perfect = 336 
    x_perfect = torch.randn(batch_size, lookback_perfect, channels)
    
    expected_patches = compute_num_patches(lookback_perfect, patch_size, stride, pad_end=True)
    out_perfect = patcher(x_perfect)
    
    assert out_perfect.shape[-1] == expected_patches, "Mismatch in Test 1!"
    print("Test 1 Passed! \n")

    print("=== TEST 2: Sequence Requiring Padding ===")
    lookback_awkward = 100 
    x_awkward = torch.randn(batch_size, lookback_awkward, channels)
    
    stride = 8
    patcher = Patching(patch_size=patch_size, stride=stride, pad_end=True)
    expected_patches_awkward = compute_num_patches(lookback_awkward, patch_size, stride=stride, pad_end=True)
    out_awkward = patcher(x_awkward)
    
    assert out_awkward.shape[-1] == expected_patches_awkward, "Mismatch in Test 2!"
    print("Test 2 Passed!\n")
    print("All tests passed successfully!")
    print("=" * 80)

def test_oracle_coattention_block():
    print("Running Modular ORACLE Co-Attention Block Tests...\n")
    
    B = 2             
    C = 7             
    lookback = 336    
    patch_size = 16
    stride = 16
    d_model = 128
    nheads = 8
    dim_feedforward = 256
    
    N = compute_num_patches(lookback, patch_size, stride, pad_end=True)
    
    oracle_block = ModularOracleCoAttentionBlock(
        d_model=d_model,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        dropout=0.1
    )

    z_t_input = torch.randn(B * C, N, d_model)
    z_f_input = torch.randn(B, C, d_model)

    z_t_out, z_f_out = oracle_block(z_t=z_t_input, z_f=z_f_input, B=B, C=C, N=N)

    assert z_t_out.shape == (B * C, N, d_model), "Temporal stream shape corrupted!"
    assert z_f_out.shape == (B, C, d_model), "Feature stream shape corrupted!"
    
    print("All dimensions align! The Modular Co-Attention Block passed tests.")
    print("=" * 80)


if __name__ == "__main__":
    # 1. Verify the math and tensor routing
    trace_oracle_dimensions()
    
    # 2. Verify the gradients, learning capacity, and visualize results
    test_oracle_learning()

    # 3. Test the patching mechanism
    test_patching()

    # 4. Test the modular co-attention block
    test_oracle_coattention_block()


'''
==================================================
ORACLE ARCHITECTURE & TENSOR TRACER
==================================================
/home/inal.mashukov001/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:128: UserWarning: CUDA initialization: CUDA driver initialization failed, you might not have a CUDA gpu. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
--- USING DEVICE: CPU ---

--- FULL ARCHITECTURE ---
ORACLE(
  (temporal_embed): TemporalEmbedding(
    (patcher): Patching()
    (W_p): Linear(in_features=2, out_features=16, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (feature_embed): FeatureEmbedding(
    (W_f): Linear(in_features=10, out_features=16, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (layers): ModuleList(
    (0): ModularOracleCoAttentionBlock(
      (time_specialist): TimeSpecialist(
        (norm_t1): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (t_self_attn): SDPA_MultiheadAttention(
          (q_proj): Linear(in_features=16, out_features=16, bias=True)
          (k_proj): Linear(in_features=16, out_features=16, bias=True)
          (v_proj): Linear(in_features=16, out_features=16, bias=True)
          (out_proj): Linear(in_features=16, out_features=16, bias=True)
        )
        (norm_t2): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (t_cross_attn): SDPA_MultiheadAttention(
          (q_proj): Linear(in_features=16, out_features=16, bias=True)
          (k_proj): Linear(in_features=16, out_features=16, bias=True)
          (v_proj): Linear(in_features=16, out_features=16, bias=True)
          (out_proj): Linear(in_features=16, out_features=16, bias=True)
        )
        (norm_t3): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (ffn_t): Sequential(
          (0): Linear(in_features=16, out_features=64, bias=True)
          (1): GELU(approximate='none')
          (2): Dropout(p=0.1, inplace=False)
          (3): Linear(in_features=64, out_features=16, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (feature_specialist): FeatureSpecialist(
        (norm_f1): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (f_self_attn): SDPA_MultiheadAttention(
          (q_proj): Linear(in_features=16, out_features=16, bias=True)
          (k_proj): Linear(in_features=16, out_features=16, bias=True)
          (v_proj): Linear(in_features=16, out_features=16, bias=True)
          (out_proj): Linear(in_features=16, out_features=16, bias=True)
        )
        (norm_f2): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (f_cross_attn): SDPA_MultiheadAttention(
          (q_proj): Linear(in_features=16, out_features=16, bias=True)
          (k_proj): Linear(in_features=16, out_features=16, bias=True)
          (v_proj): Linear(in_features=16, out_features=16, bias=True)
          (out_proj): Linear(in_features=16, out_features=16, bias=True)
        )
        (norm_f3): TemporalBatchNorm(
          (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (ffn_f): Sequential(
          (0): Linear(in_features=16, out_features=64, bias=True)
          (1): GELU(approximate='none')
          (2): Dropout(p=0.1, inplace=False)
          (3): Linear(in_features=64, out_features=16, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (norm_t_final): TemporalBatchNorm(
    (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (norm_f_final): TemporalBatchNorm(
    (bn): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (head): Linear(in_features=160, out_features=4, bias=True)
)
-------------------------

--- FORWARD PASS TRACE ---
1. Input 'x' shape: 
   torch.Size([2, 10, 2]) -> (Batch, Lookback, Channels) on cpu

2. Temporal Embedding output 'z_t' shape: 
   torch.Size([4, 9, 16]) -> (B*C, Num_Patches, d_model) on cpu
   Raw unfold 'patches' shape: 
   torch.Size([2, 2, 2, 9]) -> (B, C, Num_Patches, Patch_Size)

3. Feature Embedding output 'z_f' shape: 
   torch.Size([2, 2, 16]) -> (Batch, Channels, d_model)

4. Co-Attention Layer 0 outputs:
   'z_t' shape: torch.Size([4, 9, 16])
   'z_f' shape: torch.Size([2, 2, 16])

5. Post-Norm shapes remain identical.

6. Flattened Temporal stream 'z_t_flat' shape: 
   torch.Size([2, 2, 144]) -> (Batch, Channels, Num_Patches * d_model)

7. Joint Concatenation 'z_combined' shape: 
   torch.Size([2, 2, 160]) -> (Batch, Channels, Joint_Dim)

8. Final Output shape: 
   torch.Size([2, 2, 4]) -> (Batch, Channels, Forecast_Len) on cpu
==================================================

 Starting Toy Training Loop (Sine/Cosine Forecasting)...
--- TRAINING ON: CPU ---
Epoch [100/500], MSE Loss: 0.021489
Epoch [200/500], MSE Loss: 0.009844
Epoch [300/500], MSE Loss: 0.004113
Epoch [400/500], MSE Loss: 0.002741
Epoch [500/500], MSE Loss: 0.001856

Training complete. Generating and saving visualization...
Plot saved successfully to: model_forecast_plot.png
Running Patching Tests...

================================================================================
=== TEST 1: Perfectly Fitting Sequence ===
Test 1 Passed! 

=== TEST 2: Sequence Requiring Padding ===
Test 2 Passed!

All tests passed successfully!
================================================================================
Running Modular ORACLE Co-Attention Block Tests...

All dimensions align! The Modular Co-Attention Block passed tests.
================================================================================
'''    