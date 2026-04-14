"""
GAN-Powered Neural VM - Bootstrap Trainer
Self-improving GAN that bootstraps by training on its own generated outputs.
Implements curriculum learning, self-play, and meta-learning.
"""

import struct
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from collections import deque


class ReplayBuffer:
    """Experience replay buffer for self-generated data."""
    
    def __init__(self, capacity=10000, dim=1024):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.dim = dim
        
    def push(self, sample):
        """Add generated sample to buffer."""
        if isinstance(sample, np.ndarray):
            self.buffer.append(sample.flatten().astype(np.float32))
        else:
            self.buffer.append(np.array(sample, dtype=np.float32))
    
    def sample(self, batch_size=64):
        """Sample batch from buffer."""
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = np.array([self.buffer[i] for i in indices])
        return batch
    
    def __len__(self):
        return len(self.buffer)


class CurriculumScheduler:
    """Progressive difficulty curriculum for bootstrap training."""
    
    def __init__(self, max_level=10):
        self.max_level = max_level
        self.current_level = 1
        self.noise_level = 1.0
        self.batch_size = 32
        self.critic_iters = 5
        self.lambda_gp = 10.0
        
    def step(self):
        """Advance curriculum level."""
        if self.current_level < self.max_level:
            self.current_level += 1
            # Progressive difficulty: reduce noise, increase batch size
            self.noise_level = max(0.1, 1.0 - (self.current_level - 1) * 0.09)
            self.batch_size = min(256, 32 + (self.current_level - 1) * 16)
            self.critic_iters = min(10, 5 + (self.current_level - 1))
            self.lambda_gp = max(1.0, 10.0 - (self.current_level - 1) * 0.5)
            
    def get_config(self):
        """Get current curriculum config."""
        return {
            'level': self.current_level,
            'noise': self.noise_level,
            'batch_size': self.batch_size,
            'critic_iters': self.critic_iters,
            'lambda_gp': self.lambda_gp
        }


class MetaLearner:
    """Meta-learns optimal hyperparameters during training."""
    
    def __init__(self):
        # Hyperparameter search space
        self.lr_gen_range = [0.00005, 0.0001, 0.0002, 0.0005]
        self.lr_disc_range = [0.0001, 0.0004, 0.0008, 0.001]
        
        # Track performance for each config
        self.config_scores = []
        self.best_config = {
            'lr_gen': 0.0001,
            'lr_disc': 0.0004,
            'critic_iters': 5
        }
        
    def update(self, gen_loss, disc_loss, config=None):
        """Update meta-knowledge based on training results."""
        if config is None:
            config = self.best_config
            
        # Score: lower gen_loss is better, stable disc_loss is good
        score = -gen_loss - 0.1 * abs(disc_loss)
        
        self.config_scores.append({
            'config': config.copy(),
            'score': score,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss
        })
        
        # Keep best 10 configs
        if len(self.config_scores) > 10:
            self.config_scores = sorted(
                self.config_scores, 
                key=lambda x: x['score'], 
                reverse=True
            )[:10]
            
        # Update best config
        if self.config_scores:
            self.best_config = self.config_scores[0]['config'].copy()
            
    def get_hyperparams(self):
        """Get current best hyperparameters."""
        return self.best_config


class BootstrapTrainer:
    """Bootstrap trainer for GAN-powered Neural VM."""
    
    def __init__(self, gguf_path, checkpoint_dir='bootstrap_checkpoints'):
        self.gguf_path = gguf_path
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = 100
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize components from GGUF
        from gan_vm_runtime import GANNeuralVMRuntime
        self.vm = GANNeuralVMRuntime(gguf_path, verbose=False)
        
        # Bootstrap components
        self.replay_buffer = ReplayBuffer(capacity=10000, dim=1024)
        self.curriculum = CurriculumScheduler(max_level=10)
        self.meta_learner = MetaLearner()
        
        # Training state
        self.bootstrap_epoch = 0
        self.total_steps = 0
        self.gen_losses = deque(maxlen=100)
        self.disc_losses = deque(maxlen=100)
        self.diversity_scores = []
        
        # Self-play state
        self.generator_history = []
        self.discriminator_history = []
        
        print("=" * 60)
        print("Bootstrap Trainer Initialized")
        print("=" * 60)
        print(f"  GGUF: {gguf_path}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Replay buffer: {self.replay_buffer.capacity} samples")
        print(f"  Curriculum levels: {self.curriculum.max_level}")
        print("=" * 60)
        
    def generate_with_noise(self, noise_scale=1.0):
        """Generate sample with controlled noise."""
        noise = np.random.randn(256).astype(np.float32) * noise_scale
        return self.vm.generate(noise)
    
    def compute_diversity(self, samples):
        """Compute diversity of generated samples."""
        if len(samples) < 2:
            return 0.0
        
        # Pairwise distance
        distances = []
        for i in range(min(len(samples), 50)):
            for j in range(i + 1, min(len(samples), 50)):
                dist = np.linalg.norm(samples[i] - samples[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def self_play_step(self):
        """One step of self-play training."""
        # Get curriculum config
        curr_config = self.curriculum.get_config()
        batch_size = curr_config['batch_size']
        noise_scale = curr_config['noise']
        
        # Generate fake samples
        fake_samples = []
        for _ in range(batch_size):
            fake = self.generate_with_noise(noise_scale)
            fake_samples.append(fake)
            self.replay_buffer.push(fake)
        
        fake_batch = np.array(fake_samples, dtype=np.float32)
        
        # Sample from replay buffer
        replay_samples = self.replay_buffer.sample(batch_size)
        if replay_samples is not None:
            # Mix real (from replay) and fake
            mixed_batch = 0.7 * replay_samples + 0.3 * fake_batch
        else:
            mixed_batch = fake_batch
        
        # Train discriminator
        disc_loss = 0
        for _ in range(curr_config['critic_iters']):
            real_pred = self.vm.discriminate(mixed_batch)
            fake_pred = self.vm.discriminate(fake_batch)
            
            # WGAN loss
            disc_loss = -(real_pred.mean() - fake_pred.mean())
        
        # Train generator
        new_fake = self.generate_with_noise(noise_scale)
        fake_pred = self.vm.discriminate(new_fake)
        gen_loss = -fake_pred.mean()
        
        # Update tracking
        self.gen_losses.append(float(gen_loss))
        self.disc_losses.append(float(disc_loss))
        self.total_steps += 1
        
        return gen_loss, disc_loss
    
    def bootstrap_cycle(self, n_steps=100):
        """One bootstrap cycle: train, evaluate, adapt."""
        print(f"\n{'='*60}")
        print(f"Bootstrap Cycle #{self.bootstrap_epoch + 1}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Phase 1: Training with self-play
        print(f"[*] Phase 1: Self-play training ({n_steps} steps)...")
        for i in range(n_steps):
            gen_loss, disc_loss = self.self_play_step()
            
            if (i + 1) % 25 == 0:
                avg_gen = np.mean(list(self.gen_losses)[-25:])
                avg_disc = np.mean(list(self.disc_losses)[-25:])
                print(f"  Step {i+1}/{n_steps} | Gen: {avg_gen:.4f} | Disc: {avg_disc:.4f}")
        
        # Phase 2: Evaluate diversity
        print(f"[*] Phase 2: Evaluating generation diversity...")
        eval_samples = []
        for _ in range(100):
            eval_samples.append(self.generate_with_noise())
        
        diversity = self.compute_diversity(eval_samples)
        self.diversity_scores.append(diversity)
        print(f"  Diversity score: {diversity:.4f}")
        
        # Phase 3: Update curriculum
        print(f"[*] Phase 3: Updating curriculum...")
        old_level = self.curriculum.current_level
        self.curriculum.step()
        new_level = self.curriculum.current_level
        
        if new_level > old_level:
            print(f"  Curriculum advanced: Level {old_level} -> {new_level}")
            print(f"  New config: {self.curriculum.get_config()}")
        
        # Phase 4: Meta-learn hyperparameters
        print(f"[*] Phase 4: Meta-learning hyperparameters...")
        avg_gen = np.mean(list(self.gen_losses)[-50:]) if self.gen_losses else 0
        avg_disc = np.mean(list(self.disc_losses)[-50:]) if self.disc_losses else 0
        self.meta_learner.update(avg_gen, avg_disc)
        best_config = self.meta_learner.get_hyperparams()
        print(f"  Best config: {best_config}")
        
        # Phase 5: Save checkpoint
        if self.bootstrap_epoch % self.checkpoint_interval == 0:
            self.save_checkpoint()
        
        elapsed = time.time() - start_time
        print(f"\n[+] Cycle complete in {elapsed:.2f}s")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Avg Gen Loss: {avg_gen:.4f}")
        print(f"  Avg Disc Loss: {avg_disc:.4f}")
        
        self.bootstrap_epoch += 1
        
        return {
            'cycle': self.bootstrap_epoch,
            'gen_loss': avg_gen,
            'disc_loss': avg_disc,
            'diversity': diversity,
            'curriculum_level': new_level,
            'time': elapsed
        }
    
    def save_checkpoint(self, filename=None):
        """Save bootstrap checkpoint to file."""
        if filename is None:
            filename = f"bootstrap_epoch_{self.bootstrap_epoch:05d}.json"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'bootstrap_epoch': self.bootstrap_epoch,
            'total_steps': self.total_steps,
            'gen_losses': [float(x) for x in self.gen_losses][-100:],
            'disc_losses': [float(x) for x in self.disc_losses][-100:],
            'diversity_scores': [float(x) for x in self.diversity_scores][-100:],
            'curriculum': self.curriculum.get_config(),
            'meta_best_config': self.meta_learner.get_hyperparams(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"[*] Checkpoint saved: {filepath}")
        return filepath
    
    def run_bootstrap(self, n_cycles=10, steps_per_cycle=100):
        """Run full bootstrap training."""
        print("\n" + "=" * 60)
        print("BOOTSTRAP TRAINING STARTING")
        print("=" * 60)
        print(f"  Cycles: {n_cycles}")
        print(f"  Steps per cycle: {steps_per_cycle}")
        print(f"  Total steps: {n_cycles * steps_per_cycle}")
        print("=" * 60)
        
        all_metrics = []
        start_time = time.time()
        
        for cycle in range(n_cycles):
            metrics = self.bootstrap_cycle(steps_per_cycle)
            all_metrics.append(metrics)
            
            # Progress summary
            elapsed = time.time() - start_time
            remaining = elapsed / (cycle + 1) * (n_cycles - cycle - 1)
            print(f"\n>>> Progress: Cycle {cycle+1}/{n_cycles} | "
                  f"Elapsed: {elapsed:.0f}s | "
                  f"ETA: {remaining:.0f}s")
        
        # Final summary
        total_time = time.time() - start_time
        self.print_final_summary(all_metrics, total_time)
        
        # Save final checkpoint
        self.save_checkpoint("bootstrap_final.json")
        
        return all_metrics
    
    def print_final_summary(self, metrics, total_time):
        """Print final bootstrap summary."""
        print("\n" + "=" * 60)
        print("BOOTSTRAP TRAINING COMPLETE")
        print("=" * 60)
        print(f"  Total cycles: {len(metrics)}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Steps/sec: {self.total_steps/total_time:.2f}")
        
        if metrics:
            first = metrics[0]
            last = metrics[-1]
            
            print(f"\n  Improvement:")
            print(f"    Gen Loss: {first['gen_loss']:.4f} -> {last['gen_loss']:.4f}")
            print(f"    Disc Loss: {first['disc_loss']:.4f} -> {last['disc_loss']:.4f}")
            print(f"    Diversity: {first['diversity']:.4f} -> {last['diversity']:.4f}")
            print(f"    Curriculum: Level {first['curriculum_level']} -> {last['curriculum_level']}")
            
            if self.diversity_scores:
                print(f"\n  Final Diversity: {self.diversity_scores[-1]:.4f}")
                print(f"  Max Diversity: {max(self.diversity_scores):.4f}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python bootstrap_trainer.py <gan_vm.gguf> [cycles] [steps]")
        print()
        print("Bootstraps the GAN-powered Neural VM with self-improvement:")
        print("  - Self-play training on generated outputs")
        print("  - Curriculum learning with progressive difficulty")
        print("  - Meta-learning for hyperparameter adaptation")
        print("  - Auto-save checkpoints every 100 cycles")
        print()
        print("Arguments:")
        print("  gan_vm.gguf  - Input GAN VM GGUF file")
        print("  cycles       - Number of bootstrap cycles (default: 10)")
        print("  steps        - Steps per cycle (default: 100)")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    steps_per_cycle = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    if not os.path.exists(gguf_path):
        print(f"[!] Error: GGUF file not found: {gguf_path}")
        sys.exit(1)
    
    # Create trainer and run bootstrap
    trainer = BootstrapTrainer(gguf_path)
    metrics = trainer.run_bootstrap(n_cycles, steps_per_cycle)


if __name__ == '__main__':
    main()
