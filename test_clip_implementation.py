#!/usr/bin/env python3
"""
Test script for CLIP-style text encoder implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import models

def test_clip_text_encoder():
    """Test CLIPTextEncoder forward pass."""
    print("Testing CLIPTextEncoder...")
    
    # Create encoder
    encoder = models.CLIPTextEncoder(
        vocab_size=config.CLIP_VOCAB_SIZE,
        embed_dim=config.TEXT_EMBEDDING_DIM,
        max_length=config.MAX_TEXT_LENGTH,
        num_layers=config.CLIP_TEXT_ENCODER_LAYERS,
        num_heads=config.CLIP_TEXT_ENCODER_HEADS,
        use_clip_style=True
    )
    
    # Create dummy input (batch_size=2, sequence_length=10)
    # Using random token IDs (0-9999)
    batch_size = 2
    seq_length = 10
    dummy_tokens = torch.randint(0, config.CLIP_VOCAB_SIZE-1, (batch_size, seq_length))
    
    # Add [EOS] token at position 5 for first sample, position 8 for second
    dummy_tokens[0, 5] = config.CLIP_VOCAB_SIZE - 1  # [EOS] token
    dummy_tokens[1, 8] = config.CLIP_VOCAB_SIZE - 1  # [EOS] token
    
    # Pad to max_length
    padded_tokens = torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long)
    padded_tokens[:, :seq_length] = dummy_tokens
    
    # Forward pass
    output = encoder(padded_tokens)
    
    print(f"  Input shape: {padded_tokens.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: [{batch_size}, {config.TEXT_EMBEDDING_DIM}]")
    
    assert output.shape == (batch_size, config.TEXT_EMBEDDING_DIM), \
        f"Output shape mismatch: {output.shape}"
    
    # Test logit scale
    logit_scale = encoder.get_logit_scale()
    print(f"  Logit scale: {logit_scale.item():.4f}")
    
    print("✓ CLIPTextEncoder test passed!")

def test_image_projection():
    """Test ImageProjection forward pass."""
    print("\nTesting ImageProjection...")
    
    # Create projection head
    latent_dim = config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W
    projection = models.ImageProjection(
        latent_dim=latent_dim,
        embed_dim=config.IMAGE_PROJECTION_DIM
    )
    
    # Create dummy VAE latent (batch_size=2, channels=8, height=6, width=6)
    batch_size = 2
    dummy_latent_4d = torch.randn(batch_size, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W)
    
    # Test with 4D input
    output_4d = projection(dummy_latent_4d)
    print(f"  4D Input shape: {dummy_latent_4d.shape}")
    print(f"  4D Output shape: {output_4d.shape}")
    print(f"  Expected output shape: [{batch_size}, {config.IMAGE_PROJECTION_DIM}]")
    
    assert output_4d.shape == (batch_size, config.IMAGE_PROJECTION_DIM), \
        f"4D output shape mismatch: {output_4d.shape}"
    
    # Test with flattened input
    dummy_latent_flat = dummy_latent_4d.flatten(1)
    output_flat = projection(dummy_latent_flat)
    print(f"  Flat Input shape: {dummy_latent_flat.shape}")
    print(f"  Flat Output shape: {output_flat.shape}")
    
    # Check that both outputs are similar (allowing for small numerical differences)
    assert torch.allclose(output_4d, output_flat, rtol=1e-5), "4D and flat outputs differ!"
    
    print("✓ ImageProjection test passed!")

def test_contrastive_loss():
    """Test CLIPContrastiveLoss."""
    print("\nTesting CLIPContrastiveLoss...")
    
    # Create loss function
    loss_fn = models.CLIPContrastiveLoss(
        temperature=config.CONTRASTIVE_TEMPERATURE,
        learnable_temperature=config.LEARNABLE_TEMPERATURE
    )
    
    # Create dummy embeddings
    batch_size = 4
    embed_dim = 512
    image_emb = torch.randn(batch_size, embed_dim)
    text_emb = torch.randn(batch_size, embed_dim)
    
    # Compute loss
    loss = loss_fn(image_emb, text_emb)
    
    print(f"  Image embeddings shape: {image_emb.shape}")
    print(f"  Text embeddings shape: {text_emb.shape}")
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test temperature getter
    temperature = loss_fn.get_temperature()
    print(f"  Temperature: {temperature:.4f}")
    
    # Loss should be positive
    assert loss.item() > 0, "Loss should be positive"
    
    # Test with normalized embeddings (should still work)
    image_emb_norm = nn.functional.normalize(image_emb, dim=-1)
    text_emb_norm = nn.functional.normalize(text_emb, dim=-1)
    loss_norm = loss_fn(image_emb_norm, text_emb_norm)
    print(f"  Loss with normalized embeddings: {loss_norm.item():.4f}")
    
    print("✓ CLIPContrastiveLoss test passed!")

def test_integration():
    """Test integration of all components."""
    print("\nTesting integration of all components...")
    
    # Create all components
    text_encoder = models.CLIPTextEncoder(
        vocab_size=config.CLIP_VOCAB_SIZE,
        embed_dim=config.TEXT_EMBEDDING_DIM,
        max_length=config.MAX_TEXT_LENGTH,
        num_layers=config.CLIP_TEXT_ENCODER_LAYERS,
        num_heads=config.CLIP_TEXT_ENCODER_HEADS,
        use_clip_style=True
    )
    
    latent_dim = config.LATENT_CHANNELS * config.LATENT_H * config.LATENT_W
    image_projection = models.ImageProjection(
        latent_dim=latent_dim,
        embed_dim=config.IMAGE_PROJECTION_DIM
    )
    
    contrastive_loss = models.CLIPContrastiveLoss(
        temperature=config.CONTRASTIVE_TEMPERATURE,
        learnable_temperature=config.LEARNABLE_TEMPERATURE
    )
    
    # Create dummy data
    batch_size = 2
    
    # Text input
    seq_length = 15
    text_tokens = torch.randint(0, config.CLIP_VOCAB_SIZE-2, (batch_size, seq_length))
    # Add [EOS] tokens
    text_tokens[0, 10] = config.CLIP_VOCAB_SIZE - 1
    text_tokens[1, 12] = config.CLIP_VOCAB_SIZE - 1
    # Pad to max_length
    padded_tokens = torch.zeros(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long)
    padded_tokens[:, :seq_length] = text_tokens
    
    # Image latent
    image_latent = torch.randn(batch_size, config.LATENT_CHANNELS, config.LATENT_H, config.LATENT_W)
    
    # Forward passes
    text_emb = text_encoder(padded_tokens)
    image_emb = image_projection(image_latent)
    
    # Compute contrastive loss
    loss = contrastive_loss(image_emb, text_emb)
    
    print(f"  Text embeddings shape: {text_emb.shape}")
    print(f"  Image embeddings shape: {image_emb.shape}")
    print(f"  Contrastive loss: {loss.item():.4f}")
    
    # Check dimensions match
    assert text_emb.shape == image_emb.shape, \
        f"Text and image embedding shapes don't match: {text_emb.shape} vs {image_emb.shape}"
    
    print("✓ Integration test passed!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing CLIP-style implementation for Schrödinger Bridge")
    print("=" * 60)
    
    try:
        # Set device to CPU for testing
        config.DEVICE = torch.device('cpu')
        
        # Run tests
        test_clip_text_encoder()
        test_image_projection()
        test_contrastive_loss()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()